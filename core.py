import argparse
import csv
import io
import json
import logging
import re
import threading
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, TypeVar

warnings.filterwarnings("ignore", message=".*tpfmodel submodule.*")
import lightkurve as lk

logging.getLogger("lightkurve").setLevel(logging.ERROR)
from astroquery.mast import Catalogs, Observations
from astroquery.mast import conf as mast_conf
from astroquery.exceptions import RemoteServiceError
from astroquery.simbad import conf as simbad_conf

mast_conf.timeout = 120
simbad_conf.timeout = 120
for mast in (Observations, Catalogs):
    mast._portal_api_connection.TIMEOUT = mast_conf.timeout
    mast._service_api_connection.TIMEOUT = mast_conf.timeout
import numpy as np
import requests
from astropy.timeseries import BoxLeastSquares
from matplotlib.figure import Figure
from scipy.ndimage import median_filter

OUTPUT = Path(__file__).parent / "output"
INDEX = Path(__file__).parent / "cache.json"
TOI_MAX_AGE = 7 * 86400
DURATIONS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.25]
GRID_DURATIONS = [d for d in DURATIONS if d >= 0.04]
MIN_PERIOD = 0.2
MAX_SPAN = 6
CHUNK = 2000
MIN_SNR = 7.0
WAVE_RATIO = 0.8
VARIABILITY_FLAGS = {"too long", "wave", "unsteady"}
NOISE_FLAGS = {"too short", "one dip"}
SHORT_RATIO = 0.3
LONG_RATIO = 3.0
MIN_ORBIT = 1.5
ONE_DIP = 0.8
GIANT_RADIUS = 3.0
GIANT_MASS = 1.5
MAIN_SEQUENCE = [
    (3000, 0.15, 0.12), (3300, 0.26, 0.24), (3500, 0.40, 0.40), (3800, 0.55, 0.55), (4200, 0.68, 0.68),
    (4600, 0.75, 0.75), (5000, 0.82, 0.83), (5400, 0.88, 0.92), (5800, 1.00, 1.00), (6200, 1.15, 1.15),
    (6600, 1.40, 1.35), (7000, 1.60, 1.55), (7500, 1.85, 1.75), (8500, 2.00, 2.10), (10000, 2.30, 2.50),
]
STEADY_GAP = 1.5
STEADY_MIN_TRANSITS = 3
MAX_EARTHS = 25
SMALL_STAR = 0.7
SMALL_STAR_EARTHS = 15
RECHECK_SECTORS = 30
RECHECK_WINDOW = 0.005
RECHECK_SEASON_GAP = 60
RECHECK_SEASON = 170
RECHECK_CONTROLS = (0.79, 0.83, 0.87, 0.91, 0.94, 1.06, 1.09, 1.13, 1.17, 1.21)
NEIGHBOR_RADIUS = 120
TESS_PIXEL = 21
TESS_BLUR = 2.0
GRID_HALF = 5
PIXEL_SNR = 4.0
SATURATION_TMAG = 7.5
EXOFOP = "https://exofop.ipac.caltech.edu/tess"
TESS_EBS = "J/ApJS/258/16"
CONTAMINATION_NOTE = 0.1
FALSE_POSITIVE = {"FP", "FA", "EB", "APC"}
MAST_TRIES = 3
T = TypeVar("T")
MAST_WAIT = 30
MAST_LIMIT = 180
DOWNLOAD_LIMIT = 900


def quiet(*_):
    pass


def limited(call: Callable[[], T], limit: float) -> T:
    box: dict[str, Any] = {}

    def work():
        try:
            box["value"] = call()
        except BaseException as e:
            box["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(limit)
    if worker.is_alive():
        raise TimeoutError(f"no answer within {limit:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def retry(call: Callable[[], T], log=print, tries=MAST_TRIES, wait=MAST_WAIT, limit=MAST_LIMIT) -> T:
    for attempt in range(1, tries):
        try:
            return limited(call, limit)
        except (TimeoutError, ConnectionError, requests.RequestException, RemoteServiceError) as e:
            log(f"The archive did not answer ({type(e).__name__}) - trying again in {wait * attempt} s")
            time.sleep(wait * attempt)
    return limited(call, limit)


def load_index():
    try:
        return json.loads(INDEX.read_text())
    except (OSError, ValueError):
        return {"names": {}, "tois": {}}


def save_index(index):
    INDEX.write_text(json.dumps(index, indent=1))


def normalize(target):
    return " ".join(target.lower().split())

def resolve_tic(target, index):
    match = re.fullmatch(r"\s*tic\s*(\d+)\s*", target, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return index["names"].get(normalize(target))


def mast_dir():
    return Path(lk.config.get_cache_dir()) / "mastDownload" / "TESS"


def local_files(tic):
    if tic is None:
        return []
    return sorted(mast_dir().glob(f"*-{tic:016d}-*/*_lc.fits"))


def stitch(curves):
    return lk.LightCurveCollection(curves).stitch().remove_nans()


def sector_of(path):
    match = re.search(r"-s(\d{4})-", Path(path).name)
    return int(match.group(1)) if match else None


def compact(sectors, n, max_span=MAX_SPAN):
    sectors = sorted(sectors)
    best = []
    for i in range(len(sectors)):
        group = [x for x in sectors[i:] if x - sectors[i] <= max_span][:n]
        if (len(group), -(group[-1] - group[0]), group[-1]) > (len(best), -(best[-1] - best[0]) if best else 0, best[-1] if best else -1):
            best = group
    return best


def cadence_rank(row):
    return (str(row["author"]) != "SPOC", abs(float(row["exptime"]) - 120))


def download(target, max_sectors, log=print, progress=quiet):
    index = load_index()
    tic = resolve_tic(target, index)
    on_disk = {sector_of(f): f for f in local_files(tic) if sector_of(f) is not None}
    chosen = compact(on_disk, max_sectors)
    if len(chosen) >= max_sectors and chosen[-1] - chosen[0] <= 2 * (max_sectors - 1):
        log(f"Found sectors {', '.join(map(str, chosen))} on disk for TIC {tic} - no download needed")
        progress(1, "Loading from disk")
        lc = stitch([lk.read(on_disk[sector]) for sector in chosen])
        lc.meta["SECTORS"] = chosen
        return lc
    if on_disk:
        log(f"Have {len(on_disk)} sector(s) on disk - checking the archive for a closer-together set")

    progress(0, "Searching the TESS archive")
    results = retry(lambda: lk.search_lightcurve(target, mission="TESS", author=["SPOC", "TESS-SPOC"]), log)
    if len(results) == 0:
        raise ValueError(f"No TESS light curves found for {target}")
    table = results.table
    if table is None:
        raise ValueError(f"No TESS light curves found for {target}")
    numbers = np.asarray(table["sequence_number"]).astype(int).tolist()
    best = {}
    for i, sector in enumerate(numbers):
        if sector not in best or cadence_rank(table[i]) < cadence_rank(table[best[sector]]):
            best[sector] = i
    chosen = compact(best, max_sectors)
    log(f"Using sectors {', '.join(map(str, chosen))}")
    curves = []
    fresh = 0
    for n, sector in enumerate(chosen):
        i = best[sector]
        row = table[i]
        path = mast_dir() / str(row["obs_id"]) / str(row["productFilename"])
        if path.exists():
            progress(n / len(chosen), f"Light curve {n + 1}/{len(chosen)} already on disk")
            curves.append(lk.read(path))
        else:
            progress(n / len(chosen), f"Downloading light curve {n + 1}/{len(chosen)}")
            curves.append(retry(lambda: results[i].download(), log, limit=DOWNLOAD_LIMIT))
            fresh += 1
    log(f"Used {len(chosen) - fresh} light curve(s) from disk, downloaded {fresh}")
    progress(1, "Stitching light curves")
    lc = stitch(curves)
    lc.meta["SECTORS"] = chosen
    found = tic_id(lc)
    if found:
        index["names"][normalize(target)] = found
        save_index(index)
    return lc


def trim_gap_edges(lc, gap=0.5, trim=0.25):
    t = lc.time.value
    starts = np.concatenate([[t[0]], t[1:][np.diff(t) > gap]])
    ends = np.concatenate([t[:-1][np.diff(t) > gap], [t[-1]]])
    keep = np.ones(len(t), dtype=bool)
    for s, e in zip(starts, ends):
        keep &= ~(((t >= s) & (t < s + trim)) | ((t > e - trim) & (t <= e)))
    return lc[keep]


def remove_flares(lc, sigma=3.0):
    t = lc.time.value
    flux = np.asarray(lc.flux.value, dtype=float)
    resid = flux - median_filter(flux, size=181, mode="nearest")
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    hot = resid > sigma * noise
    edges = np.diff(np.concatenate([[0], hot.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    keep = np.ones(len(t), dtype=bool)
    flares = 0
    for a, b in zip(starts, ends):
        peak = resid[a:b].max() / noise
        if (b - a >= 2 and peak > 4) or peak > 6:
            flares += 1
            after = 0.15 if peak > 50 else 0.05
            keep &= ~((t >= t[a] - 0.01) & (t <= t[b - 1] + after))
    return lc[keep], flares


def clean(lc):
    lc, flares = remove_flares(trim_gap_edges(lc))
    flat = lc.flatten(window_length=901)
    flat = flat.remove_outliers(sigma_upper=3, sigma_lower=20).bin(time_bin_size=10 / 1440).remove_nans()
    return flat, flares


def count_transits(lc, period, t0, duration, coverage=0.5):
    return count_epochs(lc.time.value, period, t0, duration, coverage)


def count_epochs(t, period, t0, duration, coverage=0.5):
    if len(t) < 2:
        return 0
    expected = duration / np.median(np.diff(t))
    epochs = np.round((t - t0) / period)
    in_transit = np.abs(t - t0 - epochs * period) < duration / 2
    found, counts = np.unique(epochs[in_transit], return_counts=True)
    return int((counts >= max(1, coverage * expected)).sum())


def durations_for(period):
    return [d for d in DURATIONS if d < 0.3 * period] or DURATIONS[:1]


def periodogram(t, y, dy, periods, progress=quiet):
    bls = BoxLeastSquares(t, y, dy)
    power = np.empty(len(periods))
    for i in range(0, len(periods), CHUNK):
        chunk = periods[i : i + CHUNK]
        power[i : i + CHUNK] = bls.power(chunk, durations_for(chunk.min())).power
        progress(min(i + CHUNK, len(periods)) / len(periods))
    best = periods[np.argmax(power)]
    stats = bls.power([best], durations_for(best))
    return bls, power, best, stats.transit_time[0], stats.duration[0], stats.depth[0]


def search(flat, max_signals, min_sde, progress=quiet):
    t = flat.time.value
    y = np.asarray(flat.flux.value, dtype=float)
    dy = np.asarray(flat.flux_err.value, dtype=float)
    if not np.all(np.isfinite(dy)):
        dy = None
    baseline = t.max() - t.min()
    periods = BoxLeastSquares(t, y, dy).autoperiod(
        GRID_DURATIONS, minimum_period=MIN_PERIOD, maximum_period=max(baseline / 2, 1.0), frequency_factor=10
    )
    keep = np.ones(len(t), dtype=bool)
    signals = []
    for n in range(max_signals):
        if keep.sum() < 0.2 * len(t):
            break
        step = lambda f, n=n: progress((n + f) / max_signals, f"Searching for signal {n + 1}/{max_signals}")
        bls, power, period, t0, duration, depth = periodogram(
            t[keep], y[keep], None if dy is None else dy[keep], periods, step
        )
        sde = (power.max() - power.mean()) / power.std()
        if sde < min_sde:
            break
        remaining = flat[keep]
        signals.append({
            "period": period,
            "t0": t0,
            "duration": duration,
            "depth": depth,
            "sde": sde,
            "periods": periods,
            "power": power,
            "lc": remaining,
            "transits": count_transits(remaining, period, t0, duration),
        })
        keep &= ~bls.transit_mask(t, period, duration * 1.5, t0)
    progress(1, "Search done")
    return signals


def tic_id(lc):
    for key in ("TICID", "TARGETID"):
        if lc.meta.get(key):
            return int(lc.meta[key])
    return None


def known_tois(tic, log=print):
    if tic is None:
        return []
    index = load_index()
    cached = index["tois"].get(str(tic))
    if cached and time.time() - cached["time"] < TOI_MAX_AGE:
        log("Known-planet list loaded from disk")
        return cached["tois"]
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        archive: Any = NasaExoplanetArchive
        table = archive.query_criteria(
            table="toi", select="toi,pl_orbper,tfopwg_disp", where=f"tid={tic}"
        )
    except Exception as  e:
        log(f"Could not check known TOIs: {e}")
        return []
    tois = []
    for row in table:
        period = getattr(row["pl_orbper"], "value", row["pl_orbper"])
        if period is None or np.ma.is_masked(period) or not np.isfinite(period):
            continue
        tois.append({"toi": str(row["toi"]), "period": float(period), "disp": str(row["tfopwg_disp"])})
    index["tois"][str(tic)] = {"time": time.time(), "tois": tois}
    save_index(index)
    return tois


def match_known(period, tois, tolerance=0.01):
    for toi in tois:
        for ratio in (1, 2, 0.5, 3, 1 / 3):
            if abs(period - toi["period"] * ratio) / (toi["period"] * ratio) < tolerance:
                return toi, ratio
    return None, None


def same_period(a, b, tolerance=0.01):
    return any(abs(a - b * r) / (b * r) < tolerance for r in (1, 2, 0.5, 3, 1 / 3))


def window_noise(t, flux, out, dur, base):
    bins = np.floor((t[out] - t.min()) / dur).astype(int)
    counts = np.bincount(bins)
    means = np.bincount(bins, flux[out])[counts > 0] / counts[counts > 0]
    full = counts[counts > 0] >= counts.max() / 2
    return 1.4826 * np.median(np.abs(means[full] - base))


def dip(flux, mask, epoch, base, noise):
    if not mask.any():
        return None, float("inf")
    depths = [base - flux[mask & (epoch == e)].mean() for e in np.unique(epoch[mask])]
    return np.median(depths), 1.2533 * noise / np.sqrt(len(depths))


def steadiness(t, flux, inside, epoch, base, noise):
    starts = np.concatenate([[0], np.where(np.diff(t) > STEADY_GAP)[0] + 1])
    chunk = np.zeros(len(t), int)
    chunk[starts[1:]] = 1
    chunk = np.cumsum(chunk)
    worst = None
    for c in np.unique(chunk):
        here = chunk == c
        if len(np.unique(epoch[inside & here])) < STEADY_MIN_TRANSITS:
            continue
        depth, err = dip(flux, inside & here, epoch, base, noise)
        rest, rest_err = dip(flux, inside & ~here, epoch, base, noise)
        if depth is None or rest is None:
            continue
        sigma = abs(depth - rest) / np.hypot(err, rest_err)
        if abs(depth - rest) > 0.5 * max(depth, rest) and (worst is None or sigma > worst[0]):
            worst = (sigma, depth, rest, t[here].min(), t[here].max())
    return worst


def one_dip(flux, inside, epoch, base):
    found, counts = np.unique(epoch[inside], return_counts=True)
    full = found[counts >= 0.5 * counts.max()]
    if len(full) < 2:
        return None, len(full)
    depths = np.array([base - flux[inside & (epoch == e)].mean() for e in full])
    if depths.mean() <= 0:
        return None, len(full)
    return float(depths.max() / (depths.mean() * np.sqrt(len(depths)))), len(full)


def wave_ratio(phase, flux, period, dur):
    inside = np.abs(phase) < dur / 2
    if inside.sum() < 3 or (~inside).sum() < 3:
        return 0.0
    total = ((flux - flux.mean()) ** 2).sum()
    box = flux.copy()
    box[inside] = flux[inside].mean()
    box[~inside] = flux[~inside].mean()
    angle = 2 * np.pi * phase / period
    design = np.column_stack([np.ones_like(angle), np.cos(angle), np.sin(angle), np.cos(2 * angle), np.sin(2 * angle)])
    wave = design @ np.linalg.lstsq(design, flux, rcond=None)[0]
    gain_box = total - ((flux - box) ** 2).sum()
    gain_wave = total - ((flux - wave) ** 2).sum()
    return gain_wave / gain_box if gain_box > 0 else 0.0


def vet(signal, star):
    lc = signal["lc"]
    t = lc.time.value
    flux = np.asarray(lc.flux.value, dtype=float)
    period, t0, dur = signal["period"], signal["t0"], signal["duration"]
    phase = (t - t0 + period / 2) % period - period / 2
    epoch = np.round((t - t0) / period).astype(int)
    out = np.abs(phase) > dur
    flags = []
    signal["snr"] = None
    if dur / period > 0.2:
        flags.append(("too long", f"each dip lasts {dur / period:.0%} of the orbit - too long for a planet, more like a star that brightens and fades"))
    if star["radius"] and signal["depth"] > 0:
        earths = np.sqrt(signal["depth"]) * star["radius"] * 109.1
        if earths > MAX_EARTHS:
            flags.append(("too big", f"the dip means an object {earths:.0f}x Earth's width - bigger than any known planet (the largest are ~22x), so it is a star"))
        elif star["radius"] < SMALL_STAR and earths > SMALL_STAR_EARTHS:
            flags.append(("too big", f"the dip means an object {earths:.0f}x Earth's width around a small star ({star['radius']:.2f}x the Sun) - giant planets around small stars stay under ~{SMALL_STAR_EARTHS}x, a small red dwarf companion is this size"))
    elif signal["depth"] > 0.3:
        flags.append(("too big", f"the star loses {signal['depth']:.0%} of its light - no planet blocks that much, a second star does"))
    flags += orbit_flags(signal, star)
    if out.sum() < 50:
        return flags
    ratio = wave_ratio(phase, flux, period, dur)
    signal["wave_ratio"] = ratio
    if ratio >= WAVE_RATIO:
        flags.append(("wave", f"the folded light curve is a smooth wave, not a sharp dip (a wave explains {ratio:.0%} as much as a transit shape) - the star's own spots or pulsations"))
    base = np.median(flux[out])
    noise = window_noise(t, flux, out, dur, base)
    half = window_noise(t, flux, out, dur / 2, base)
    inside = np.abs(phase) < dur / 2
    if inside.any() and noise > 0:
        depths = [base - flux[inside & (epoch == e)].mean() for e in np.unique(epoch[inside])]
        signal["snr"] = np.mean(depths) * np.sqrt(len(depths)) / noise
        share, n = one_dip(flux, inside, epoch, base)
        signal["one_dip"] = share
        if share is not None and share > ONE_DIP:
            flags.append(("one dip", f"one single dip carries most of the signal ({share:.0%} of what all {n} together give, vs. {1 / np.sqrt(n):.0%} if they were equal) - one glitch or stellar event, not a repeating transit"))
        worst = steadiness(t, flux, np.abs(phase) < dur / 4, epoch, base, half)
        if worst is not None and worst[0] > 3:
            sigma, depth, rest, start, end = worst
            flags.append((
                "unsteady",
                f"the dip is {depth * 1e6:.0f} ppm deep in days {start:.0f}-{end:.0f} but {rest * 1e6:.0f} ppm in the rest of the data ({sigma:.1f} sigma) - "
                "a planet blocks the same light every orbit; spots, flares of activity or instrument trouble come and go",
            ))

    odd, odd_err = dip(flux, inside & (epoch % 2 == 1), epoch, base, noise)
    even, even_err = dip(flux, inside & (epoch % 2 == 0), epoch, base, noise)
    if odd is not None and even is not None:
        diff = abs(odd - even) / np.hypot(odd_err, even_err)
        if diff > 3:
            flags.append(("odd/even", f"every other dip has a different depth ({diff:.1f} sigma) - two stars eclipsing each other"))

    core, core_err = dip(flux, np.abs(phase) < dur / 4, epoch, base, half)
    if core is not None:
        signal["core_depth"], signal["core_err"] = core, core_err
    wing, _ = dip(flux, inside & (np.abs(phase) >= dur / 4), epoch, base, half)
    if core is not None and wing is not None and core > 5 * core_err and wing / core < 0.5:
        flags.append(("V-shape", "the dip is a sharp V instead of a flat-bottomed U - typical of two stars grazing each other"))

    second, second_err = dip(flux, np.abs(np.abs(phase) - period / 2) < dur / 2, np.round((t - t0) / period - 0.5), base, noise)
    if second is not None and second > 3 * second_err and (second > 1e-3 or second > 0.15 * signal["depth"]):
        flags.append(("secondary", f"a second, smaller dip halfway between the main ones ({second / second_err:.1f} sigma) - a planet hides behind its star without a visible dip, a second star does not"))
    return flags


PLOT_BG = "#18142E"
PLOT_INK = "#C9C2E0"
PLOT_LINE = "#3B3260"
PLOT_GOLD = "#FFC46B"
PLOT_VIOLET = "#9D86FF"


def style_axes(ax):
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_INK, which="both")
    for spine in ax.spines.values():
        spine.set_color(PLOT_LINE)
    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        text.set_color(PLOT_INK)
    ax.xaxis.get_offset_text().set_color(PLOT_INK)
    ax.yaxis.get_offset_text().set_color(PLOT_INK)


def draw_pixels(fig, spot, pixels):
    best = pixels["best"]
    left, right = fig.add_subplot(spot[0]), fig.add_subplot(spot[1])
    left.imshow(best["image"], origin="lower", cmap="magma")
    limit = np.nanmax(np.abs(best["diff"])) or 1
    right.imshow(best["diff"], origin="lower", cmap="berlin", vmin=-limit, vmax=limit)
    where = f"sectors {best['sector']} combined, north up, east left" if best.get("combined") else f"sector {best['sector']}"
    for ax, title in ((left, f"Star image, {where}"), (right, f"Light lost during the dips ({best['transits']} transits)")):
        for x, y, dmag in best["neighbors"]:
            ax.scatter(x, y, s=max(10, 80 - 12 * dmag), facecolors="none", edgecolors="#B9A8E8", lw=0.8)
        ax.scatter(*best["target"], s=160, facecolors="none", edgecolors="#6CD4E0", lw=2)
        ax.scatter(*best["source"], s=120, marker="+", color=PLOT_GOLD, lw=2)
        ax.set_title(title)
        ax.set_xlabel("TESS pixels" if best.get("combined") else "Pixel column")
        ax.set_ylabel("TESS pixels" if best.get("combined") else "Pixel row")
        style_axes(ax)
    return [left, right]


def make_figure(target, raw, signal, index, label):
    pixels = signal.get("pixels")
    pixels = pixels if pixels and "best" in pixels else None
    fig = Figure(figsize=(11, 17 if pixels else 12), facecolor=PLOT_BG, layout="constrained")
    grid = fig.add_gridspec(4 if pixels else 3, 2, height_ratios=[1, 1, 1, 1.4] if pixels else None)
    axes = [fig.add_subplot(grid[row, :]) for row in range(3)]
    if pixels:
        draw_pixels(fig, (grid[3, 0], grid[3, 1]), pixels)
    fig.suptitle(f"{target}  signal {index}  {label}", fontsize=13, color="#ECE8F7")

    axes[0].scatter(raw.time.value, raw.flux.value, s=1, c=raw.time.value, cmap="cool")
    axes[0].set_title("Raw light curve")
    axes[0].set_xlabel("Time [BTJD]")
    axes[0].set_ylabel("Flux")

    axes[1].plot(signal["periods"], signal["power"], color=PLOT_VIOLET, lw=0.6)
    axes[1].axvline(signal["period"], color=PLOT_GOLD, alpha=0.6, lw=3)
    axes[1].set_xscale("log")
    axes[1].set_title(f"BLS periodogram  (SDE {signal['sde']:.1f})")
    axes[1].set_xlabel("Period [days]")
    axes[1].set_ylabel("Power")

    folded = signal["lc"].fold(period=signal["period"], epoch_time=signal["t0"])
    phase_hours = folded.time.value * 24
    window = signal["duration"] * 24 * 3
    keep = np.abs(phase_hours) < window
    points = axes[2].scatter(
        phase_hours[keep],
        folded.flux.value[keep],
        s=3,
        c=folded.time_original.value[keep],
        cmap="cool",
    )
    edges = np.arange(-window, window + 1e-9, signal["duration"] * 24 / 8)
    flux = np.asarray(folded.flux.value[keep], dtype=float)
    which = np.digitize(phase_hours[keep], edges)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.array([flux[which == b].mean() if np.any(which == b) else np.nan for b in range(1, len(edges))])
    axes[2].plot(centers, means, color=PLOT_GOLD, lw=2)
    axes[2].set_title(
        f"Folded  P={signal['period']:.5f} d  depth={signal['depth'] * 1e6:.0f} ppm  "
        f"duration={signal['duration'] * 24:.2f} h"
    )
    axes[2].set_xlabel("Hours from mid-transit")
    axes[2].set_ylabel("Normalized flux")
    bar = fig.colorbar(points, ax=axes[2], label="Observation time [BTJD]")
    bar.outline.set_edgecolor(PLOT_LINE)
    bar.ax.tick_params(colors=PLOT_INK)
    bar.ax.yaxis.label.set_color(PLOT_INK)

    axes[0].ticklabel_format(axis="y", useOffset=False)
    axes[2].ticklabel_format(axis="y", useOffset=False)
    for ax in axes:
        style_axes(ax)
    return fig


def star_info(lc, flares):
    def num(key):
        value = lc.meta.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    return {
        "tic": tic_id(lc),
        "radius": num("RADIUS"),
        "teff": num("TEFF"),
        "tmag": num("TESSMAG"),
        "logg": num("LOGG"),
        "ra": num("RA_OBJ"),
        "dec": num("DEC_OBJ"),
        "sectors": lc.meta.get("SECTORS"),
        "flares": flares,
    }


def tic_params(star, log=print):
    if star["tic"] is None:
        return None
    index = load_index()
    cached = index.setdefault("tic", {}).get(str(star["tic"]))
    if cached:
        return cached
    try:
        catalogs: Any = Catalogs
        table = retry(lambda: catalogs.query_criteria(catalog="Tic", ID=star["tic"]), log)
    except Exception as e:
        log(f"Could not look up the star in the TESS Input Catalog: {e}")
        return None
    if len(table) == 0:
        return None
    row = table[0]

    def num(key):
        value = row[key]
        return None if np.ma.is_masked(value) or value is None or not np.isfinite(value) else float(value)

    result = {
        "radius": num("rad"), "mass": num("mass"), "rho": num("rho"), "logg": num("logg"),
        "teff": num("Teff"), "giant": str(row["lumclass"]) == "GIANT",
    }
    index["tic"][str(star["tic"])] = result
    save_index(index)
    return result


def density(star, tic):
    tic = tic or {}
    radius = star["radius"] or tic.get("radius")
    logg = star["logg"] or tic.get("logg")
    teff = star["teff"] or tic.get("teff")
    if tic.get("rho") and tic.get("radius") and (star["radius"] is None or abs(tic["radius"] - star["radius"]) < 0.05 * star["radius"]):
        return tic["rho"], "catalog"
    if radius and logg:
        return 10 ** (logg - 4.438) / radius, "catalog"
    if radius:
        return radius ** 1.25 / radius ** 3, "radius"
    if tic.get("giant"):
        return GIANT_MASS / GIANT_RADIUS ** 3, "giant"
    if teff:
        temps, radii, masses = zip(*MAIN_SEQUENCE)
        r = np.interp(teff, temps, radii)
        return float(np.interp(teff, temps, masses) / r ** 3), "temperature"
    return None, None


def add_tic(star, log=print):
    tic = tic_params(star, log)
    if tic:
        star["radius"] = star["radius"] or tic["radius"]
        star["teff"] = star["teff"] or tic["teff"]
        star["logg"] = star["logg"] or tic["logg"]
    star["giant"] = bool(tic and tic["giant"])
    star["density"], star["density_source"] = density(star, tic)
    return star


def orbit_size(period, rho):
    return 4.2 * period ** (2 / 3) * rho ** (1 / 3)


def central_duration(period, rho, depth):
    a = orbit_size(period, rho)
    return period / np.pi * np.arcsin(min(1.0, (1 + np.sqrt(max(depth, 0))) / a))


def orbit_flags(signal, star):
    rho = star.get("density")
    if not rho:
        return []
    source = star["density_source"]
    period, dur = signal["period"], signal["duration"]
    a = orbit_size(period, rho)
    expected = central_duration(period, rho, signal["depth"])
    signal["orbit_size"], signal["expected_duration"] = a, expected
    how = {
        "catalog": "", "radius": " (density guessed from its size)",
        "giant": " (catalogued as a giant star; assumed only 3x the Sun's width, so the real orbit is even tighter)",
        "temperature": " (density guessed from its temperature, assuming an ordinary star like the Sun)",
    }[source]
    flags = []
    if a < MIN_ORBIT:
        flags.append(("inside star", f"an orbit of {period * 24:.1f} hours would be only {a:.1f}x the star's radius from its center{how} - inside or grazing the star, so no planet can orbit there; it is two stars or a pulsation"))
    elif dur < SHORT_RATIO * expected:
        flags.append(("too short", f"each dip lasts {dur * 24:.1f} h, but a planet crossing this star every {period:.1f} days would take about {expected * 24:.1f} h{how} - too short to be a transit, more like a glitch or noise lining up"))
    elif source == "catalog" and dur > LONG_RATIO * expected:
        flags.append(("too long for star", f"each dip lasts {dur * 24:.1f} h, but a planet crossing this star every {period:.1f} days would take at most about {expected * 24:.1f} h - the dip probably comes from a bigger star blended into the pixels, or the star is not what the catalog says"))
    return flags


def gaia_neighbors(star, log=print):
    if star["tic"] is None or star["ra"] is None or star["dec"] is None:
        return None
    index = load_index()
    cached = index.setdefault("neighbors", {}).get(str(star["tic"]))
    if cached and all(len(s) >= 4 for s in cached["others"]):
        return cached
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        center = SkyCoord(star["ra"], star["dec"], unit="deg")
        vizier: Any = Vizier(columns=["RA_ICRS", "DE_ICRS", "RAJ2000", "DEJ2000", "Gmag", "RPmag"], row_limit=-1)
        tables = vizier.query_region(center, radius=NEIGHBOR_RADIUS * u.arcsec, catalog="I/355/gaiadr3")
    except Exception as e:
        log(f"Could not check nearby stars: {e}")
        return None
    stars = []
    if tables:
        table = tables[0]
        ra = np.where(np.ma.getmaskarray(table["RAJ2000"]), table["RA_ICRS"], table["RAJ2000"])
        dec = np.where(np.ma.getmaskarray(table["DEJ2000"]), table["DE_ICRS"], table["DEJ2000"])
        seps = np.atleast_1d(center.separation(SkyCoord(ra, dec, unit="deg")).arcsec)
        for row, sep, r, d in zip(table, seps, ra, dec):
            mag = row["RPmag"] if not np.ma.is_masked(row["RPmag"]) else row["Gmag"]
            if not np.ma.is_masked(mag):
                stars.append([float(sep), float(mag), float(r), float(d)])
    close = [s for s in stars if s[0] < 10]
    if not close:
        log("Could not find the star itself in Gaia - skipping the nearby-star check")
        return None
    guess = star["tmag"] if star["tmag"] is not None else min(s[1] for s in close)
    target = min(close, key=lambda s: abs(s[1] - guess) + s[0] / 10)
    result = {"mag": target[1], "others": sorted(s for s in stars if s is not target)}
    index["neighbors"][str(star["tic"])] = result
    save_index(index)
    return result


def blend_check(signal, neighbors):
    suspects = []
    for sep, mag, *_ in neighbors["others"]:
        leak = np.exp(-((sep / TESS_PIXEL) ** 2) / (2 * TESS_BLUR**2))
        ratio = 10 ** (-0.4 * (mag - neighbors["mag"])) * leak
        needed = signal["depth"] * (1 + ratio) / ratio
        if needed < 1:
            suspects.append({"sep": sep, "dmag": mag - neighbors["mag"], "needed": needed})
    return sorted(suspects, key=lambda x: x["needed"])


VSX_WORDS = {
    "EA": "eclipsing binary", "EB": "eclipsing binary", "EW": "eclipsing binary", "E": "eclipsing binary",
    "ELL": "ellipsoidal binary (two stars so close they stretch each other)",
    "ROT": "spotted rotating star", "BY": "spotted rotating star", "RS": "spotted rotating star",
    "DSCT": "pulsating star", "HADS": "pulsating star", "GDOR": "pulsating star", "RRAB": "pulsating star",
    "RRC": "pulsating star", "SR": "pulsating giant", "L": "slowly varying giant", "M": "pulsating giant",
    "UV": "flare star", "EP": "star with a transiting planet",
}
BINARY_KINDS = {"eclipsing binary", "ellipsoidal binary (two stars so close they stretch each other)"}
SIMBAD_WORDS = {
    "EB*": "an eclipsing binary", "SB*": "a spectroscopic binary (two stars seen in its light)",
    "**": "a double star", "RotV*": "a spotted rotating star", "BYDra": "a spotted rotating star",
    "RSCVn": "a spotted rotating star", "Pu*": "a pulsating star", "dS*": "a pulsating star",
    "gD*": "a pulsating star", "RRLyr": "a pulsating star", "Er*": "a flare star", "Fl*": "a flare star",
    "V*": "a variable star", "WD*": "a white dwarf", "YSO": "a very young star",
}


def harmonic(period, catalog_period, tolerance=0.003):
    for n in range(1, 6):
        for m in range(1, 6):
            if abs(period - catalog_period * n / m) / (catalog_period * n / m) < tolerance:
                return n, m
    return None


def vsx_kind(vtype):
    for token in re.split(r"[/+:|()]", vtype):
        if token in VSX_WORDS:
            return VSX_WORDS[token]
    return f"variable star (type {vtype})"


def catalog_check(star, log=print):
    if star["tic"] is None or star["ra"] is None or star["dec"] is None:
        return None
    index = load_index()
    cached = index.setdefault("catalogs", {}).get(str(star["tic"]))
    if cached and "tessebs" in cached:
        return cached
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad
        from astroquery.vizier import Vizier

        center = SkyCoord(star["ra"], star["dec"], unit="deg")
        vizier: Any = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "Type", "Period"], row_limit=-1)
        tables = vizier.query_region(center, radius=NEIGHBOR_RADIUS * u.arcsec, catalog="B/vsx/vsx")
        ebs: Any = Vizier(columns=["TIC", "RAJ2000", "DEJ2000", "Per"], row_limit=-1)
        eb_tables = ebs.query_region(center, radius=NEIGHBOR_RADIUS * u.arcsec, catalog=TESS_EBS)
        simbad = Simbad()
        simbad.add_votable_fields("otype")
        nearby = simbad.query_region(center, radius=5 * u.arcsec)
    except Exception as e:
        log(f"Could not check star catalogs: {e}")
        return None
    vsx = []
    if tables:
        table = tables[0]
        seps = np.atleast_1d(center.separation(SkyCoord(table["RAJ2000"], table["DEJ2000"], unit="deg")).arcsec)
        for row, sep in zip(table, seps):
            period = None if np.ma.is_masked(row["Period"]) else float(row["Period"])
            vsx.append([float(sep), str(row["Name"]), str(row["Type"]), period])
    tessebs = []
    if eb_tables:
        table = eb_tables[0]
        seps = np.atleast_1d(center.separation(SkyCoord(table["RAJ2000"], table["DEJ2000"], unit="deg")).arcsec)
        for row, sep in zip(table, seps):
            period = None if np.ma.is_masked(row["Per"]) else float(row["Per"])
            tessebs.append([float(sep), int(row["TIC"]), period])
    types = []
    if nearby is not None:
        for row in nearby:
            otype = str(row["otype"])
            if not otype.startswith("Pl"):
                types.append([str(row["main_id"]), otype])
    result = {"vsx": sorted(vsx), "tessebs": sorted(tessebs), "simbad": types}
    index["catalogs"][str(star["tic"])] = result
    save_index(index)
    return result


def read_catalogs(signal, catalogs):
    relabel, notes = None, []
    for sep, name, vtype, period in catalogs["vsx"]:
        kind = vsx_kind(vtype)
        here = sep < 10
        where = f"this star ({name})" if here else f"a star {sep:.0f}\" away ({name})"
        cycle = f"period {period:.4f} d" if period else "no period listed"
        match = harmonic(signal["period"], period) if period else None
        if kind in BINARY_KINDS and match:
            relabel = relabel or ("catalogued binary (VSX)" if here else "catalogued binary nearby (VSX)")
            same = "the same period" if match == (1, 1) else f"a matching period (this signal is {match[0]}/{match[1]} of it)"
            notes.append(f"VSX lists {where} as an {kind} with {same}: {period:.4f} d - that is almost certainly what made this dip.")
        elif kind in BINARY_KINDS and (here or sep < 60):
            notes.append(f"VSX lists {where} as an {kind} ({cycle}). Different period from this signal, but be careful.")
        elif here and kind == VSX_WORDS["EP"]:
            notes.append(f"VSX lists this star ({name}) as a star with a known transiting planet ({cycle}) - compare the periods.")
        elif here:
            notes.append(f"VSX lists this star ({name}) as a {kind} ({cycle}). Its own brightness changes can fake or hide dips.")
    for sep, tic, period in catalogs.get("tessebs", []):
        here = sep < 10
        where = "this star" if here else f"a star {sep:.0f}\" away (TIC {tic})"
        match = harmonic(signal["period"], period) if period else None
        if match:
            relabel = relabel or ("catalogued binary (TESS EB)" if here else "catalogued binary nearby (TESS EB)")
            same = "the same period" if match == (1, 1) else f"a matching period (this signal is {match[0]}/{match[1]} of it)"
            notes.append(f"The TESS eclipsing binary catalog lists {where} with {same}: {period:.4f} d - that is almost certainly what made this dip.")
        elif here or sep < 60:
            cycle = f"period {period:.4f} d" if period else "no period listed"
            notes.append(f"The TESS eclipsing binary catalog lists {where} ({cycle}). Different period from this signal, but be careful.")
    for main_id, otype in catalogs["simbad"]:
        words = SIMBAD_WORDS.get(otype)
        if not words:
            continue
        if otype == "EB*":
            relabel = relabel or "catalogued binary (SIMBAD)"
        notes.append(f"SIMBAD lists this star ({main_id}) as {words}.")
    return relabel, list(dict.fromkeys(notes))


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Shadowcatch"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        return reply.read().decode("utf-8", errors="replace")


def ctoi_list(log=print):
    cached = load_index().get("ctois")
    if cached and time.time() - cached["time"] < TOI_MAX_AGE:
        return cached["stars"]
    try:
        text = fetch(f"{EXOFOP}/download_ctoi.php?sort=ctoi&output=csv")
    except Exception as e:
        log(f"Could not download the ExoFOP community candidate list: {e}")
        return cached["stars"] if cached else None
    stars = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            period = float(row.get("Period (days)") or "nan")
        except ValueError:
            period = float("nan")
        if not row.get("TIC ID") or not row.get("CTOI"):
            continue
        period = period if np.isfinite(period) and period > 0 else None
        disp = row.get("TFOPWG Disposition") or row.get("User Disposition") or ""
        stars.setdefault(row["TIC ID"], []).append([row["CTOI"], period, disp, row.get("Promoted to TOI", ""), row.get("User", "")])
    if not stars:
        log("The ExoFOP community candidate list came back empty")
        return cached["stars"] if cached else None
    index = load_index()
    index["ctois"] = {"time": time.time(), "stars": stars}
    save_index(index)
    log(f"Loaded {sum(map(len, stars.values()))} community candidates (CTOIs) from ExoFOP")
    return stars


def exofop_target(star, log=print):
    if star["tic"] is None:
        return None
    index = load_index()
    cached = index.setdefault("exofop", {}).get(str(star["tic"]))
    if cached and time.time() - cached["time"] < TOI_MAX_AGE:
        return cached["info"]
    try:
        page = json.loads(fetch(f"{EXOFOP}/target.php?id={star['tic']}&json"))
    except Exception as e:
        log(f"Could not read this star's ExoFOP page: {e}")
        return None
    basic = page.get("basic_info") or {}
    try:
        contamination = float(basic.get("tic_contamination_ratio") or "nan")
    except ValueError:
        contamination = float("nan")
    info = {
        "contamination": contamination if np.isfinite(contamination) else None,
        "confirmed": basic.get("confirmed_planets") or "",
        "companions": len(page.get("stellar_companions") or []),
        "imaging": len(page.get("imaging") or []),
        "spectra": len(page.get("spectroscopy") or []),
        "light_curves": len(page.get("time_series") or []),
        "notes": [
            f"{t.get('toi', '?')} ({t.get('disp_tfop') or t.get('disp_tess') or 'no disposition'}): {t['notes'].strip()}"
            for t in page.get("tois") or [] if (t.get("notes") or "").strip()
        ],
    }
    index = load_index()
    index.setdefault("exofop", {})[str(star["tic"])] = {"time": time.time(), "info": info}
    save_index(index)
    return info


def exofop_check(star, log=print):
    if star["tic"] is None:
        return None
    stars = ctoi_list(log)
    info = exofop_target(star, log)
    if stars is None and info is None:
        return None
    return {"ctois": None if stars is None else stars.get(str(star["tic"]), []), "info": info}


def read_exofop(signal, exofop):
    relabel, notes = None, []
    ctois = exofop["ctois"]
    if ctois:
        known = [{"toi": c[0], "period": c[1], "disp": c[2]} for c in ctois if c[1] is not None]
        match, ratio = match_known(signal["period"], known)
        if match is not None:
            bad = f", marked {match['disp']}" if match["disp"] in FALSE_POSITIVE else ""
            relabel = f"known CTOI {match['toi']}{bad}" if ratio == 1 else f"alias of CTOI {match['toi']} (x{ratio:g})"
            notes.append(
                f"Someone already posted this signal on ExoFOP as community candidate CTOI {match['toi']} "
                f"(period {match['period']:.5f} d{bad})."
            )
        for c in ctois:
            if match is None or c[0] != match["toi"]:
                what = f"period {c[1]:.5f} d" if c[1] is not None else "a single dip, no period yet"
                notes.append(f"ExoFOP has another community candidate on this star: CTOI {c[0]}, {what} ({c[2] or 'no disposition'}) - compare its dip times with yours.")
    elif ctois is None:
        notes.append("Could not download the ExoFOP community candidate list.")
    info = exofop["info"]
    if info:
        if info["confirmed"]:
            notes.append(f"ExoFOP lists confirmed planet(s) here: {info['confirmed']}.")
        notes += [f"ExoFOP note on {note}" for note in info["notes"]]
        if info["contamination"] is not None and info["contamination"] >= CONTAMINATION_NOTE:
            notes.append(
                f"The TESS star catalog estimates other stars add {info['contamination']:.0%} on top of this star's own light "
                "in its pixels - real dips look shallower, and a faint eclipsing neighbor is easier to hide."
            )
        if info["companions"]:
            notes.append(f"ExoFOP lists {info['companions']} close stellar companion(s) found with sharp imaging.")
        followup = [f"{n} {what}" for n, what in ((info["imaging"], "imaging"), (info["spectra"], "spectra"), (info["light_curves"], "ground-based light curves")) if n]
        if followup:
            notes.append(f"Astronomers have already observed this star: {', '.join(followup)} on ExoFOP.")
    return relabel, notes


def recheck(signal, star, log=print, flat=None):
    if star["tic"] is None:
        return None
    used = set(star.get("sectors") or [])
    try:
        found = retry(lambda: lk.search_lightcurve(f"TIC {star['tic']}", mission="TESS", author=["SPOC", "TESS-SPOC"]), log)
    except Exception as e:
        log(f"Could not search for other sectors: {e}")
        return None
    table = found.table
    if table is None or len(found) == 0:
        return {"verdict": "no other data", "sectors": []}
    numbers = np.asarray(table["sequence_number"]).astype(int).tolist()
    best = {}
    for i, sector in enumerate(numbers):
        if sector not in used and (sector not in best or cadence_rank(table[i]) < cadence_rank(table[best[sector]])):
            best[sector] = i
    if not best:
        log("No other sectors to check the signal in")
        return {"verdict": "no other data", "sectors": []}
    chosen = sorted(best)[-RECHECK_SECTORS:]
    log(f"Checking the signal in other sectors ({', '.join(map(str, chosen))})")
    curves = []
    for sector in chosen:
        try:
            lc = retry(lambda: found[best[sector]].download(quality_bitmask="default"), log, limit=DOWNLOAD_LIMIT)
        except Exception as e:
            log(f"Could not download sector {sector}: {e}")
            continue
        if lc is not None:
            curves.append(lc)
    if not curves:
        return None
    flat_other = stitch([clean(lc.normalize())[0] for lc in curves])
    t = flat_other.time.value
    flux = np.asarray(flat_other.flux.value, dtype=float)
    err = np.asarray(flat_other.flux_err.value, dtype=float)
    period, dur, expected = signal["period"], signal["duration"], signal["depth"]
    bls = BoxLeastSquares(t, flux, err)
    def strict(center):
        step = min(center * dur / max(np.ptp(t), 1) / 4, 2 * RECHECK_WINDOW * center / 2000)
        return bls.power(np.arange(center * (1 - RECHECK_WINDOW), center * (1 + RECHECK_WINDOW), step), [dur])

    result = strict(period)
    k = int(np.argmax(result.power))
    best_period, t0 = float(result.period[k]), float(result.transit_time[k])
    transits = count_transits(flat_other, best_period, t0, dur)
    report = {"sectors": chosen, "transits": transits, "expected": expected}
    if transits < 2:
        return {**report, "verdict": "too little data"}
    phase = (t - t0 + best_period / 2) % best_period - best_period / 2
    out = np.abs(phase) > dur
    noise = window_noise(t, flux, out, dur, np.median(flux[out]))
    depth = float(result.depth[k])
    depth_err = max(float(result.depth_err[k]), noise / np.sqrt(transits))
    report.update(depth=depth, depth_err=depth_err)
    controls = []
    for factor in RECHECK_CONTROLS:
        control = strict(period * factor)
        controls.append(float(control.depth[int(np.argmax(control.power))]))
    floor = float(np.median(controls) + 3 * 1.4826 * np.median(np.abs(np.array(controls) - np.median(controls))))
    report.update(floor=floor)
    loose = season_depth(t, flux, err, period, dur)
    if loose is not None:
        report.update(loose_depth=loose[0], loose_err=loose[1])
    if flat is not None:
        joined = stitch([flat, flat_other])
        report.update(ephemeris=joint_ephemeris(joined, signal))
    if expected - depth > 3 * depth_err and depth < 0.5 * expected and loose is not None and loose[0] < 0.75 * expected:
        return {**report, "verdict": "missing"}
    if expected < floor and transits >= 3:
        return {**report, "verdict": "chance level"}
    if depth > 3 * depth_err and depth > report["floor"] and depth >= 0.5 * expected and transits >= 3:
        return {**report, "verdict": "seen again"}
    if depth < floor and expected - depth > 3 * depth_err and transits >= 3:
        return {**report, "verdict": "too weak"}
    return {**report, "verdict": "unclear"}


def joint_ephemeris(lc, signal):
    t = lc.time.value
    flux = np.asarray(lc.flux.value, dtype=float)
    err = np.asarray(lc.flux_err.value, dtype=float)
    period, t0, dur = signal["period"], signal["t0"], signal["duration"]
    step = min(period * dur / max(np.ptp(t), 1) / 4, 2 * RECHECK_WINDOW * period / 2000)
    result = BoxLeastSquares(t, flux, err).power(np.arange(period * (1 - RECHECK_WINDOW), period * (1 + RECHECK_WINDOW), step), [dur])
    k = int(np.argmax(result.power))
    best, when = float(result.period[k]), float(result.transit_time[k])
    epoch = when + np.round((t0 - when) / best) * best
    if abs(epoch - t0) > dur:
        return None
    return [best, float(epoch)]


def season_depth(t, flux, err, period, dur):
    edges = [0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > RECHECK_SEASON_GAP or t[i] - t[edges[-1]] > RECHECK_SEASON:
            edges.append(i)
    edges.append(len(t))
    fits = []
    for a, b in zip(edges[:-1], edges[1:]):
        tb, fb, eb = t[a:b], flux[a:b], err[a:b]
        if len(tb) < 100:
            continue
        step = min(period * dur / max(np.ptp(tb), 1) / 4, 2 * RECHECK_WINDOW * period / 500)
        result = BoxLeastSquares(tb, fb, eb).power(np.arange(period * (1 - RECHECK_WINDOW), period * (1 + RECHECK_WINDOW), step), [dur])
        k = int(np.argmax(result.power))
        n = count_epochs(tb, float(result.period[k]), float(result.transit_time[k]), dur)
        if n < 2:
            continue
        phase = (tb - result.transit_time[k] + result.period[k] / 2) % result.period[k] - result.period[k] / 2
        out = np.abs(phase) > dur
        noise = window_noise(tb, fb, out, dur, np.median(fb[out]))
        fits.append((float(result.depth[k]), max(float(result.depth_err[k]), noise / np.sqrt(n))))
    if not fits:
        return None
    weights = np.array([1 / e ** 2 for _, e in fits])
    depths = np.array([d for d, _ in fits])
    return float((weights * depths).sum() / weights.sum()), float(1 / np.sqrt(weights.sum()))


def recheck_words(check):
    if check is None:
        return "Could not download other TESS sectors to check the signal (no connection)."
    sectors = ", ".join(map(str, check["sectors"]))
    if check["verdict"] == "no other data":
        return "TESS has no other sectors of this star, so the signal can't be checked in independent data yet."
    if check["verdict"] == "too little data":
        return f"The other sectors ({sectors}) caught fewer than 2 dips at this period - not enough to check."
    measured = f"{check['depth'] * 1e6:.0f} +- {check['depth_err'] * 1e6:.0f} ppm in sectors {sectors} ({check['transits']} dips) vs {check['expected'] * 1e6:.0f} ppm here"
    if "floor" in check:
        measured += f" (the same search at 10 unrelated periods says anything under ~{check['floor'] * 1e6:.0f} ppm can be chance)"
    if "loose_depth" in check:
        measured += f"; {check['loose_depth'] * 1e6:.0f} +- {check['loose_err'] * 1e6:.0f} ppm letting each half-year find its own timing"
    if check["verdict"] == "missing":
        return f"Missing in other TESS data: {measured}. A planet doesn't switch off - this was noise or the star's own wobbles lining up."
    if check["verdict"] == "chance level":
        return (
            f"Chance-sized: {measured}. Even with far more data, folding at random periods turns up dips this deep, "
            "so a dip this shallow can't be told apart from noise."
        )
    if check["verdict"] == "too weak":
        return (
            f"Too weak in other TESS data: {measured}. The other sectors would have shown a dip this deep, "
            "but what they show is no bigger than chance and well short of the expected depth."
        )
    if check["verdict"] == "seen again":
        return f"Seen again in independent TESS data: {measured}. Good sign - a real repeating dip."
    return f"Other TESS data doesn't settle it (too noisy, or the two ways of measuring disagree): {measured}."


def ephemeris(signal):
    t = signal["lc"].time.value
    period, dur = signal["period"], signal["duration"]
    snr = signal.get("snr") or 7.0
    epochs = np.round((t - signal["t0"]) / period)
    seen = np.unique(epochs[np.abs(t - signal["t0"] - epochs * period) < dur / 2])
    middle = seen[np.argmin(np.abs(seen - np.median(seen)))] if len(seen) else 0
    t0_err = dur / (2 * snr)
    span = max(seen.max() - seen.min(), 1) if len(seen) else 1
    return signal["t0"] + middle * period, t0_err, t0_err * np.sqrt(12) / span


def ctoi(signal, star):
    period, dur, depth = signal["period"], signal["duration"], signal["depth"]
    snr = signal.get("snr") or 7.0
    t0, t0_err, period_err = ephemeris(signal)
    step = min((abs(d - dur) for d in DURATIONS if d != dur), default=dur)
    dur_err = max(dur / snr, step / 2)
    if signal.get("core_depth") and signal["core_depth"] > depth:
        depth, depth_err = signal["core_depth"], max(signal["core_err"], signal["core_depth"] / snr)
    else:
        depth_err = depth / snr
    fields = {
        "TIC ID": star["tic"],
        "Transit Epoch (BJD)": f"{t0 + 2457000:.5f}",
        "Transit Epoch (BJD) err": f"{t0_err:.5f}",
        "Period (days)": f"{period:.6f}",
        "Period (days) Error": f"{period_err:.6f}",
        "Depth ppm": f"{depth * 1e6:.0f}",
        "Depth ppm Error": f"{depth_err * 1e6:.0f}",
        "Duration (hrs)": f"{dur * 24:.2f}",
        "Duration (hrs) Error": f"{dur_err * 24:.2f}",
        "Planet Radius (R_Earth)": "",
        "Planet Radius (R_Earth) Error": "",
        "User Disposition": "PC",
        "Notes": ctoi_notes(signal, star),
    }
    if star["radius"] and depth > 0:
        earths = np.sqrt(depth) * star["radius"] * 109.1
        fields["Planet Radius (R_Earth)"] = f"{earths:.2f}"
        fields["Planet Radius (R_Earth) Error"] = f"{earths * np.hypot(0.5 * depth_err / depth, 0.07):.2f}"
    return fields


def ctoi_notes(signal, star):
    sectors = star.get("sectors")
    parts = [
        f"Found with Shadowcatch: BLS on SPOC/TESS-SPOC light curves"
        + (f" (sectors {', '.join(map(str, sectors))})" if sectors else ""),
        f"SDE {signal['sde']:.1f}",
        f"SNR {signal['snr']:.1f}" if signal.get("snr") is not None else "SNR not measured",
        f"{signal['transits']} transits",
        "passed odd/even, secondary eclipse, V-shape, duration, size, single-transit and stellar-density checks",
    ]
    if signal.get("expected_duration"):
        parts.append(
            f"expected central-transit duration {signal['expected_duration'] * 24:.1f} h for a circular orbit "
            f"(stellar density from {star['density_source']}), a/R* ~{signal['orbit_size']:.1f}"
        )
    blends = signal.get("blends")
    if blends is None:
        parts.append("Gaia blend check not run")
    elif blends:
        worst = blends[0]
        parts.append(
            f"{len(blends)} Gaia neighbor(s) within {NEIGHBOR_RADIUS}\" could cause the signal as a blended EB "
            f"(most likely: {worst['sep']:.0f}\" away, dTmag~{worst['dmag']:.1f}, would need {worst['needed']:.{0 if worst['needed'] >= 0.1 else 1}%} eclipse)"
        )
    else:
        parts.append(f"no Gaia neighbor within {NEIGHBOR_RADIUS}\" bright enough to cause it")
    exofop = signal.get("exofop")
    if exofop is None:
        parts.append("ExoFOP CTOI list not checked")
    elif exofop:
        parts.extend(exofop)
    else:
        parts.append("no CTOI or ExoFOP notes for this TIC")
    catalog = signal.get("catalog")
    if catalog is None:
        parts.append("VSX/SIMBAD not checked")
    elif catalog:
        parts.extend(catalog)
    else:
        parts.append("not listed as variable or binary in VSX/SIMBAD")
    check = signal.get("recheck")
    if check is None:
        parts.append("independent-sector check not run")
    elif check["verdict"] in ("seen again", "unclear"):
        parts.append(
            f"independent sectors {', '.join(map(str, check['sectors']))}: depth {check['depth'] * 1e6:.0f} +- "
            f"{check['depth_err'] * 1e6:.0f} ppm over {check['transits']} transits ({check['verdict']})"
        )
    else:
        parts.append(f"independent-sector check: {check['verdict']}")
    pixels = signal.get("pixels")
    if pixels is None:
        parts.append("pixel-level check not run")
    elif pixels["verdict"] == "no data":
        parts.append("no SPOC target pixel files for a difference-image check")
    elif pixels["verdict"] == "saturated":
        parts.append("target saturated, difference-image check not possible")
    else:
        best = pixels["best"]
        parts.append(
            f"difference image ({len(pixels['sectors'])} sector(s), best S{best['sector']}): {pixels['verdict']}, "
            f"source offset {best['offset'] * TESS_PIXEL:.0f}\" from the target's light centroid, peak {best['snr']:.1f} sigma"
        )
    parts.append("depth from the central half of the transit; uncertainties approximate (from SNR, not a transit model fit; radius assumes 7% stellar radius error)")
    return "; ".join(part.rstrip(".") for part in parts)


def ctoi_text(target, signal, star):
    fields = ctoi(signal, star)
    width = max(map(len, fields))
    lines = [
        f"Community TOI values for {target}, signal {signal['index']}",
        "Field names match ExoFOP's CTOI table. ExoFOP only accepts candidates already published in a peer-reviewed journal, uploaded by approved users - share these values with a researcher who can follow it up.",
        "",
    ]
    lines += [f"{name.ljust(width)}  {value}" for name, value in fields.items() if name != "Notes"]
    lines += ["", "Notes:", fields["Notes"], "", "Pipe-separated (same order):", "|".join(fields), "|".join(str(v) for v in fields.values())]
    return "\n".join(lines) + "\n"


def centroid(image, x, y, size=1):
    ny, nx = image.shape
    x0, x1 = max(x - size, 0), min(x + size + 1, nx)
    y0, y1 = max(y - size, 0), min(y + size + 1, ny)
    box = np.clip(image[y0:y1, x0:x1], 0, None)
    total = box.sum()
    if total <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((box * xx).sum() / total), float((box * yy).sum() / total)


def stacked_difference(t, flux, period, center, dur):
    epoch = np.round((t - center) / period)
    phase = t - center - epoch * period
    diffs, outs = [], []
    for e in np.unique(epoch):
        mine = epoch == e
        inside = mine & (np.abs(phase) < dur / 2)
        around = mine & (np.abs(phase) > dur) & (np.abs(phase) < 2.5 * dur)
        if inside.sum() >= 3 and around.sum() >= 5:
            out = flux[around].mean(axis=0)
            diffs.append(out - flux[inside].mean(axis=0))
            outs.append(out)
    return diffs, outs


def difference_image(tpf, signal):
    t = tpf.time.value
    flux = np.asarray(tpf.flux.value, dtype=float)
    good = np.isfinite(t) & np.all(np.isfinite(flux.reshape(len(t), -1)), axis=1)
    t, flux = t[good], flux[good]
    period, t0, dur = signal["period"], signal["t0"], signal["duration"]
    diffs, outs = stacked_difference(t, flux, period, t0, dur)
    if not diffs:
        return None
    nulls = []
    for frac in np.linspace(0.1, 0.9, 17):
        shift = frac * period
        if min(shift, period - shift) < 3.5 * dur:
            continue
        fake, _ = stacked_difference(t, flux, period, t0 + shift, dur)
        if fake:
            nulls.append(np.mean(fake[: len(diffs)], axis=0))
    if len(nulls) >= 4:
        noise = np.std(nulls, axis=0, ddof=1)
    elif len(diffs) >= 2:
        noise = np.std(diffs, axis=0, ddof=1) / np.sqrt(len(diffs))
    else:
        return None
    return outs, np.mean(diffs, axis=0), noise, len(diffs)


def pixel_neighbors(tpf, neighbors):
    if not neighbors:
        return []
    from astropy.coordinates import SkyCoord

    marks = []
    ny, nx = tpf.flux.shape[1:]
    for sep, mag, *where in neighbors["others"]:
        if len(where) < 2 or mag - neighbors["mag"] > 6:
            continue
        x, y = tpf.wcs.world_to_pixel(SkyCoord(where[0], where[1], unit="deg"))
        if -0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5:
            marks.append((float(x), float(y), mag - neighbors["mag"]))
    return marks


def saturated(image):
    top = image >= 0.9 * np.nanmax(image)
    return top.sum(axis=0).max() >= 3


def locate(image, diff, noise, allowed):
    guess = np.unravel_index(np.nanargmax(np.where(allowed, image, -np.inf)), image.shape)
    tx, ty = centroid(np.nan_to_num(image), guess[1], guess[0])
    snr = np.where(np.isfinite(noise) & (noise > 0), np.nan_to_num(diff) / np.where(noise > 0, noise, 1), 0)
    py, px = np.unravel_index(np.argmax(snr), snr.shape)
    dx, dy = centroid(np.nan_to_num(diff), px, py)
    return (tx, ty), (dx, dy), float(snr[py, px]), float(np.hypot(dx - tx, dy - ty))


def sky_grid(center):
    import astropy.units as u

    steps = (np.arange(2 * GRID_HALF + 1) - GRID_HALF) * TESS_PIXEL
    north, west = np.meshgrid(steps, steps, indexing="ij")
    return center.spherical_offsets_by(-west * u.arcsec, north * u.arcsec)


def combine_sectors(results, star, neighbors):
    from astropy.coordinates import SkyCoord
    from scipy.ndimage import map_coordinates

    center = SkyCoord(star["ra"], star["dec"], unit="deg")
    grid = sky_grid(center)
    images, diffs, weights = [], [], []
    for r in results:
        x, y = r["wcs"].world_to_pixel(grid)
        sample = lambda a: map_coordinates(a, [y, x], order=1, cval=np.nan)
        images.append(sample(r["image"] / np.nanmax(r["image"])))
        diffs.append(sample(r["diff"]))
        weights.append(1 / sample(r["noise"]) ** 2)
    weights = np.where(np.isfinite(weights), weights, 0)
    total = weights.sum(axis=0)
    diff = np.where(total > 0, (np.nan_to_num(diffs) * weights).sum(axis=0) / np.where(total > 0, total, 1), np.nan)
    noise = np.where(total > 0, 1 / np.sqrt(np.where(total > 0, total, 1)), np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        image = np.nanmean(images, axis=0)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    allowed = np.hypot(xx - GRID_HALF, yy - GRID_HALF) <= 2
    target, source, snr, offset = locate(image, diff, noise, allowed)
    marks = []
    for sep, mag, *where in (neighbors or {}).get("others", []):
        if len(where) < 2 or mag - neighbors["mag"] > 6:
            continue
        east, north = center.spherical_offsets_to(SkyCoord(where[0], where[1], unit="deg"))
        col, row = GRID_HALF - east.to_value("arcsec") / TESS_PIXEL, GRID_HALF + north.to_value("arcsec") / TESS_PIXEL
        if -0.5 <= col <= 2 * GRID_HALF + 0.5 and -0.5 <= row <= 2 * GRID_HALF + 0.5:
            marks.append((col, row, mag - neighbors["mag"]))
    return {
        "sector": "+".join(str(r["sector"]) for r in results), "transits": sum(r["transits"] for r in results),
        "snr": snr, "offset": offset, "image": image, "diff": diff, "target": target, "source": source,
        "neighbors": marks, "combined": True,
    }


def pixel_check(signal, star, log=print, max_sectors=4, neighbors=None, enough=6):
    if star["tic"] is None:
        return None
    if star["tmag"] is not None and star["tmag"] < SATURATION_TMAG:
        log("Star is too bright for the pixel check (saturated pixels)")
        return {"verdict": "saturated", "sectors": []}
    try:
        found = retry(lambda: lk.search_targetpixelfile(f"TIC {star['tic']}", mission="TESS", author="SPOC", exptime=120), log)
    except Exception as e:
        log(f"Could not search for pixel data: {e}")
        return None
    if len(found) == 0:
        log("No 2-min pixel data for this star - skipping the pixel check")
        return {"verdict": "no data", "sectors": []}
    if found.table is None:
        return {"verdict": "no data", "sectors": []}
    numbers = np.asarray(found.table["sequence_number"]).astype(int).tolist()
    wanted = set(star.get("sectors") or [])
    order = sorted(range(len(found)), key=lambda i: (numbers[i] not in wanted, -numbers[i]))
    ephemeris = (signal.get("recheck") or {}).get("ephemeris")
    if ephemeris:
        signal = {**signal, "period": ephemeris[0], "t0": ephemeris[1]}
    results = []
    for i in order[:max_sectors]:
        if sum(r["transits"] for r in results) >= enough and any(r["snr"] >= PIXEL_SNR for r in results):
            break
        sector = numbers[i]
        log(f"Getting pixel data for sector {sector} (pixel check)")
        tpf = None
        for fresh in (False, True):
            try:
                tpf = limited(lambda: found[i].download(quality_bitmask="default", cache=not fresh), DOWNLOAD_LIMIT)
                if tpf is None:
                    raise ValueError("the archive returned no file")
                tpf.flux.shape
                break
            except Exception as e:
                tpf = None
                log(f"Pixel data for sector {sector} {'failed again' if fresh else 'is damaged or missing, downloading it again'}: {e}")
        if tpf is None:
            continue
        made = difference_image(tpf, signal)
        if made is None:
            continue
        outs, diff, noise, transits = made
        image = np.mean(outs, axis=0)
        if saturated(image):
            log(f"Sector {sector} pixels are saturated - skipping the pixel check")
            return {"verdict": "saturated", "sectors": []}
        target, source, snr, offset = locate(image, diff, noise, tpf.pipeline_mask)
        results.append({
            "sector": sector, "transits": transits, "snr": snr, "offset": offset, "image": image, "diff": diff,
            "noise": noise, "wcs": tpf.wcs, "target": target, "source": source, "neighbors": pixel_neighbors(tpf, neighbors),
        })
    if not results:
        return {"verdict": "no data", "sectors": []}
    judged = results
    if len(results) >= 2 and star["ra"] is not None and star["dec"] is not None:
        judged = [combine_sectors(results, star, neighbors)]
    usable = [r for r in judged if r["snr"] >= PIXEL_SNR]
    if not usable:
        verdict = "inconclusive"
    elif all(r["offset"] < 1.0 for r in usable):
        verdict = "on target"
    elif all(r["offset"] >= 1.5 for r in usable):
        verdict = "off target"
    else:
        verdict = "unclear"
    best = max(judged, key=lambda r: r["snr"])
    close = [b for b in (signal.get("blends") or []) if b["sep"] < 1.5 * TESS_PIXEL]
    if verdict == "on target" and close:
        verdict = "too close to tell"
    return {"verdict": verdict, "sectors": results, "best": best, "close": close}


def pixel_words(pixels):
    if pixels is None:
        return "Could not run the pixel check (no connection to the TESS archive)."
    verdict = pixels["verdict"]
    if verdict == "no data":
        return "TESS has no 2-minute pixel pictures of this star, so the pixel check could not run."
    if verdict == "saturated":
        return "This star is so bright it overflows TESS's pixels (the light spills along a column), so the pixels can't show where the dip is. A ground-based telescope is needed instead."
    best = pixels["best"]
    where = f"{best['offset']:.1f} pixels ({best['offset'] * TESS_PIXEL:.0f}\") from this star, sector {best['sector']}"
    if verdict == "on target":
        return f"Good sign: TESS's pictures show the light going missing right on this star ({where}). The gold + sits in the cyan circle."
    if verdict == "too close to tell":
        near = min(pixels["close"], key=lambda b: b["sep"])
        return (
            f"The light goes missing at this star's position ({where}), but a star only {near['sep']:.0f}\" away "
            f"({near['dmag']:+.1f} magnitudes) is closer than one TESS pixel and bright enough to cause the dip. "
            "TESS can't tell them apart - a ground-based telescope can."
        )
    if verdict == "off target":
        return f"Warning: the light goes missing {where} - from somewhere else, not this star."
    if verdict == "unclear":
        return f"Mixed result: the light loss is {where}, and the sectors disagree. A ground-based telescope would settle it."
    return f"Inconclusive: each pixel is too noisy to see the dip clearly (strongest pixel {best['snr']:.1f} sigma, need {PIXEL_SNR:g})."


def size_class(earths):
    if earths < 1.6:
        return "probably rocky, like Earth"
    if earths < 4:
        return "a super-Earth or mini-Neptune"
    if earths < 8:
        return "Neptune-sized"
    if earths < 20:
        return "a gas giant like Jupiter or Saturn"
    return "too big for a planet - probably two stars eclipsing each other"


def star_kind(teff):
    if teff is None:
        return "unknown type"
    if teff < 3900:
        return "a cool red dwarf"
    if teff < 5300:
        return "an orange dwarf, a bit cooler than the Sun"
    if teff < 6000:
        return "a Sun-like star"
    if teff < 7500:
        return "a hot white-yellow star"
    return "a very hot blue-white star"


def describe(signal, star):
    lines = [f"Signal {signal['index']}: {signal['label']}", ""]

    period = signal["period"]
    if period < 1:
        orbit = f"{period * 24:.1f} hours"
    else:
        orbit = f"{period:.2f} days"
    lines.append(f"ORBIT  The dip repeats every {orbit}. If it is a planet, that is its year. (Mercury: 88 days, Earth: 365.)")

    depth = signal["depth"]
    lines.append(f"DIP    The star gets {depth * 100:.3f}% dimmer ({depth * 1e6:.0f} ppm = parts per million) during each dip.")

    if star["radius"] and depth > 0:
        earths = np.sqrt(depth) * star["radius"] * 109.1
        lines.append(
            f"SIZE   Estimated {earths:.1f}x Earth's width - {size_class(earths)}. "
            f"(Earth 1, Neptune 3.9, Jupiter 11.2.) Bigger planet = deeper dip."
        )
    else:
        lines.append("SIZE   Unknown - NASA has no size for this star, so the planet size can't be estimated.")

    lines.append(f"LENGTH Each dip lasts {signal['duration'] * 24:.1f} hours - the time the planet takes to cross the star.")
    if signal.get("expected_duration"):
        lines.append(
            f"       A planet on this orbit would take up to about {signal['expected_duration'] * 24:.1f} hours to cross this star "
            "(shorter if it only clips the edge)."
        )
    lines.append(f"COUNT  {signal['transits']} separate dips were caught in the data.")
    strength = "strong" if signal["sde"] >= 12 else "moderate" if signal["sde"] >= 9 else "weak"
    lines.append(
        f"SDE    {signal['sde']:.1f} ({strength}). How much the signal stands out from random noise. "
        "Below 7 is ignored; above ~12 is clearly not chance."
    )
    if signal.get("snr") is not None:
        lines.append(
            f"SNR    {signal['snr']:.1f}. How many times bigger the dip is than the star's own flickering "
            f"over the same few hours. Real planets need about {MIN_SNR:g} or more."
        )
    if "exofop" in signal:
        if signal["exofop"] is None:
            lines.append("EXOFOP Could not reach ExoFOP (the TESS follow-up website) to check for earlier reports.")
        elif not signal["exofop"]:
            lines.append("EXOFOP Nobody has posted a community candidate for this star on ExoFOP, and it has no notes there.")
        else:
            for note in signal["exofop"]:
                lines.append(f"EXOFOP {note}")
    if "catalog" in signal:
        if signal["catalog"] is None:
            lines.append("CATALOGS Could not check the VSX and SIMBAD star catalogs (no connection).")
        elif not signal["catalog"]:
            lines.append("CATALOGS Neither VSX (variable stars) nor SIMBAD lists this star as a binary or variable.")
        else:
            for note in signal["catalog"]:
                lines.append(f"CATALOGS {note}")
    if "blends" in signal:
        blends = signal["blends"]
        if blends is None:
            lines.append("NEIGHBORS Could not check nearby stars (no connection to the Gaia star catalog).")
        elif not blends:
            lines.append(
                f"NEIGHBORS Good sign: no star within {NEIGHBOR_RADIUS / 60:g} arcminutes is bright enough to fake "
                "this dip, so it really comes from this star."
            )
        else:
            worst = blends[0]
            lines.append(
                f"NEIGHBORS {len(blends)} nearby star(s) could fake this dip if they were two stars eclipsing each other. "
                f"TESS pixels are big ({TESS_PIXEL}\" wide), so their light mixes with this star's. Most likely: "
                f"{worst['sep']:.0f}\" away ({worst['sep'] / TESS_PIXEL:.1f} pixels), {worst['dmag']:.1f} magnitudes fainter - "
                f"it would only have to dim by {worst['needed']:.{0 if worst['needed'] >= 0.1 else 1}%}. Ground-based telescopes with sharper vision can "
                "tell which star the dip is on."
            )
    if "recheck" in signal:
        lines.append(f"RECHECK {recheck_words(signal['recheck'])}")
    if "pixels" in signal:
        lines.append(f"PIXELS {pixel_words(signal['pixels'])}")
    lines.append("")

    label = signal["label"]
    if label.startswith("known CTOI"):
        lines.append(
            "VERDICT  Already reported. Someone posted this exact signal on ExoFOP as a community candidate (CTOI), "
            "so it is not new - but your search found it independently."
            + (" They have since marked it a false positive." if label.endswith(tuple(f"marked {d}" for d in FALSE_POSITIVE)) else "")
        )
    elif label.startswith("alias of CTOI"):
        lines.append(
            "VERDICT  Same as a community candidate already on ExoFOP, but the search locked onto a multiple of its "
            "period. Not new."
        )
    elif label.startswith("known"):
        lines.append(
            "VERDICT  Already known. NASA lists this one as a TESS Object of Interest. "
            "Your search found it on its own - that means the search works."
        )
    elif label.startswith("alias"):
        lines.append(
            "VERDICT  Same as a known planet, but the search locked onto a multiple of its real period "
            "(e.g. every 2nd dip). Not new."
        )
    elif label.startswith("repeat"):
        lines.append(
            "VERDICT  The same rhythm as an earlier signal, found again because removing the first one "
            "did not remove all of it. Usually a star that changes brightness by itself. Not new."
        )
    elif label.startswith("likely variable star"):
        lines.append("VERDICT  Probably the star itself, not a planet:")
        for _, reason in signal["flags"]:
            lines.append(f"  - {reason}")
    elif label.startswith("likely binary"):
        lines.append("VERDICT  Probably two stars orbiting each other, not a planet:")
        for _, reason in signal["flags"]:
            lines.append(f"  - {reason}")
    elif label.startswith("likely blend"):
        lines.append(
            "VERDICT  Probably not a planet on this star. TESS's own pictures show the light going missing "
            "from a spot next to it - most likely a fainter star behind or beside it that is really two stars "
            "eclipsing each other. Look at the bottom plots: the gold + is where the light was lost, the cyan "
            "circle is this star."
        )
    elif label.startswith("catalogued binary"):
        lines.append(
            "VERDICT  Not a planet. Astronomers already know this is two stars eclipsing each other - "
            "your search found their eclipses, which shows it works."
        )
    elif label.startswith("not confirmed"):
        lines.append(
            "VERDICT  Probably not real. TESS watched this star in other sectors too, and the dip is not there "
            "at the depth it should have. Real planets show up every time; noise and stellar wobbles don't."
            if "missing" in label or "too weak" in label else
            "VERDICT  Probably not real. The dip is no deeper than what random noise produces in this star's other "
            "TESS data, so there is nothing to tell it apart from chance."
        )
    elif label.startswith("likely noise (SNR"):
        lines.append(
            f"VERDICT  Probably not real. The dip is only {signal['snr']:.1f} times bigger than the star's own "
            f"flickering over the same few hours (a planet needs about {MIN_SNR:g}). Restless stars - spotted, "
            "flaring or pulsating - wobble enough to fake shallow dips like this."
        )
    elif label.startswith("likely noise"):
        lines.append(
            f"VERDICT  Probably not real. Only {signal['transits']} dips - a planet needs at least 3 matching dips "
            "to be convincing. Two dips can line up by coincidence, or come from glitches, flares or data gaps."
        )
    else:
        lines.append("VERDICT  Not in NASA's list! Before getting excited, check the plots:")
        lines.append("  1. Bottom plot: do all the colors sit on top of each other? Every dip must look the same.")
        lines.append("  2. Top plot: is a dip right next to a gap, flare or jump? Then it's probably junk.")
        lines.append("  3. Shape: planets make a U or flat-bottomed dip. A sharp V often means two stars.")
        lines.append("  4. Size above ~20x Earth means two stars, not a planet.")
        lines.append("  Passes all four? Worth showing to a researcher - ExoFOP only takes published candidates.")
    return "\n".join(lines)


def describe_star(star):
    lines = [f"STAR  TIC {star['tic']} - {star_kind(star['teff'])}"]
    if star["teff"]:
        lines.append(f"  Surface temperature {star['teff']:.0f} K (Sun: 5772 K)")
    if star["radius"]:
        lines.append(f"  Width {star['radius']:.2f}x the Sun")
    if star["tmag"]:
        lines.append(f"  Brightness to TESS: magnitude {star['tmag']:.1f} (lower = brighter, less noisy data)")
    if star["flares"]:
        lines.append(f"  {star['flares']} flare(s) detected and removed before the search")
    return "\n".join(lines)


def label_for(signal, tois, earlier):
    toi, ratio = match_known(signal["period"], tois)
    if toi is not None and ratio == 1:
        return f"known TOI {toi['toi']}"
    if toi is not None:
        return f"alias of TOI {toi['toi']} (x{ratio:g})"
    for other in earlier:
        if same_period(signal["period"], other["period"]):
            return f"repeat of signal {other['index']}"
    if signal["transits"] < 3:
        return f"likely noise ({signal['transits']} transits)"
    if signal["snr"] is not None and signal["snr"] < MIN_SNR:
        return f"likely noise (SNR {signal['snr']:.1f})"
    if signal["flags"]:
        names = ", ".join(name for name, _ in signal["flags"])
        others = [name for name, _ in signal["flags"] if name not in NOISE_FLAGS]
        if not others:
            return f"likely noise ({names})"
        if all(name in VARIABILITY_FLAGS for name in others):
            return f"likely variable star ({names})"
        return f"likely binary ({names})"
    return "NEW CANDIDATE"


def run(target, sectors=10, max_signals=3, min_sde=7.0, log=print, progress=quiet):
    stage = lambda lo, hi: lambda f, text="": progress(lo + (hi - lo) * f, text)

    raw = download(target, sectors, log, stage(0, 0.3))
    stage(0.3, 0.35)(0, "Cleaning light curve")
    flat, flares = clean(raw)
    star = add_tic(star_info(raw, flares), log)
    tic = star["tic"]
    log(f"TIC {tic}: {len(flat)} points over {np.ptp(flat.time.value):.1f} days")
    if flares:
        log(f"Removed {flares} flare(s)")

    stage(0.35, 0.4)(0, "Checking known planets")
    tois = known_tois(tic, log)
    for toi in tois:
        log(f"Known TOI {toi['toi']}: P={toi['period']:.5f} d ({toi['disp']})")

    log("Searching for transits...")
    signals = search(flat, max_signals, min_sde, stage(0.4, 0.95))
    if not signals:
        log("No significant signals found")

    folder = OUTPUT / target.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    plots = stage(0.95, 1)
    neighbors = False
    catalogs = False
    exofop = False
    for i, signal in enumerate(signals, 1):
        plots((i - 1) / len(signals), f"Drawing plot {i}/{len(signals)}")
        signal["index"] = i
        signal["flags"] = vet(signal, star)
        signal["label"] = label_for(signal, tois, signals[: i - 1])
        if signal["label"] == "NEW CANDIDATE":
            if exofop is False:
                exofop = exofop_check(star, log)
            relabel, signal["exofop"] = (None, None) if exofop is None else read_exofop(signal, exofop)
            signal["label"] = relabel or signal["label"]
        if signal["label"] == "NEW CANDIDATE":
            if catalogs is False:
                catalogs = catalog_check(star, log)
            relabel, signal["catalog"] = (None, None) if catalogs is None else read_catalogs(signal, catalogs)
            signal["label"] = relabel or signal["label"]
        if signal["label"] == "NEW CANDIDATE":
            plots((i - 1) / len(signals), "Checking the signal in other TESS sectors")
            signal["recheck"] = recheck(signal, star, log, flat)
            if signal["recheck"] and signal["recheck"]["verdict"] == "missing":
                signal["label"] = "not confirmed (missing in other sectors)"
            elif signal["recheck"] and signal["recheck"]["verdict"] == "chance level":
                signal["label"] = "not confirmed (chance-sized dip)"
            elif signal["recheck"] and signal["recheck"]["verdict"] == "too weak":
                signal["label"] = "not confirmed (too weak in other sectors)"
        if signal["label"] == "NEW CANDIDATE":
            if neighbors is False:
                neighbors = gaia_neighbors(star, log)
            signal["blends"] = None if neighbors is None else blend_check(signal, neighbors)
            plots((i - 1) / len(signals), "Downloading TESS pixel images for the pixel check")
            signal["pixels"] = pixel_check(signal, star, log, neighbors=neighbors or None)
            if signal["pixels"] and signal["pixels"]["verdict"] == "off target":
                signal["label"] = "likely blend (dip is off target)"
        signal["summary"] = describe(signal, star)
        signal["figure"] = make_figure(target, raw, signal, i, signal["label"])
        signal["path"] = folder / f"signal{i}.png"
        signal["figure"].savefig(signal["path"], dpi=120)
        if signal["label"] == "NEW CANDIDATE":
            signal["ctoi"] = folder / f"signal{i}_ctoi.txt"
            signal["ctoi"].write_text(ctoi_text(target, signal, star))
        log(
            f"[{i}] P={signal['period']:.5f} d  depth={signal['depth'] * 1e6:.0f} ppm  "
            f"dur={signal['duration'] * 24:.2f} h  SDE={signal['sde']:.1f}  N={signal['transits']}  {signal['label']}  -> {signal['path']}"
        )
    progress(1, "Done")
    return {"tic": tic, "star": star, "raw": raw, "tois": tois, "signals": signals}


def main():
    parser = argparse.ArgumentParser(description="Shadowcatch: search TESS light curves for transiting planets")
    parser.add_argument("target", help='Star name or TIC ID, e.g. "TIC 261136679" or "Pi Mensae"')
    parser.add_argument("--sectors", type=int, default=10)
    parser.add_argument("--signals", type=int, default=3)
    parser.add_argument("--min-sde", type=float, default=7.0)
    args = parser.parse_args()

    result = run(args.target, args.sectors, args.signals, args.min_sde)
    print("\n" + describe_star(result["star"]))
    for signal in result["signals"]:
        print("\n" + signal["summary"])


if __name__ == "__main__":
    main()
