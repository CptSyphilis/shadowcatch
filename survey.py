import argparse
import csv
import io
import os
import subprocess
import sys
import time
import traceback

import core
import pick


def save_survey(tic, entry):
    index = core.load_index()
    index.setdefault("surveyed", {})[str(tic)] = entry
    core.save_index(index)


HEADER = ["tic", "signal", "period_d", "depth_ppm", "duration_h", "sde", "transits", "snr", "neighbors_that_could_fake_it", "catalog_notes", "pixel_check", "plot", "shared_period_with", "exofop_notes", "other_sectors", "gaia_notes"]
SHARED_TOLERANCE = 0.03
SIGNAL_TOLERANCE = 0.002
NOT_SHARED = ("known", "alias of", "repeat of")
MIN_SCORE = 3.0
SHARED_COLUMN = HEADER.index("shared_period_with")
LOG = core.OUTPUT / "survey.log"
PAUSE_FILE = core.OUTPUT / "survey.paused"


class Tee:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def log_to_file():
    core.OUTPUT.mkdir(exist_ok=True)
    log = open(LOG, "a", buffering=1)
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            same = os.path.samestat(os.fstat(stream.fileno()), os.stat(LOG))
        except (OSError, ValueError, io.UnsupportedOperation):
            same = False
        if not same:
            setattr(sys, name, Tee(stream, log))


def append_candidates(tic, signals):
    path = core.OUTPUT / "candidates.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(HEADER)
        for s in signals:
            writer.writerow([
                tic, s["index"], f"{s['period']:.5f}", f"{s['depth'] * 1e6:.0f}",
                f"{s['duration'] * 24:.2f}", f"{s['sde']:.1f}", s["transits"], "" if s.get("snr") is None else f"{s['snr']:.1f}",
                "unchecked" if s.get("blends") is None else len(s["blends"]),
                "unchecked" if s.get("catalog") is None else " | ".join(s["catalog"]),
                "unchecked" if s.get("pixels") is None else s["pixels"]["verdict"], s["path"], "",
                "unchecked" if s.get("exofop") is None else " | ".join(s["exofop"]),
                "unchecked" if s.get("recheck") is None else s["recheck"]["verdict"],
                "unchecked" if s.get("gaia") is None else " | ".join(s["gaia"]),
            ])


def signal_periods(signals):
    return [round(s["period"], 5) for s in signals if s["transits"] >= 3 and not s["label"].startswith(NOT_SHARED)]


def close(a, b, tolerance):
    return abs(a - b) / a < tolerance


def same_sector_signals(tic, period, surveyed, tolerance=SIGNAL_TOLERANCE):
    sectors = set(surveyed.get(tic, {}).get("sectors", []))
    return {
        other for other, entry in surveyed.items()
        if other != tic and sectors & set(entry.get("sectors", []))
        and any(close(period, p, tolerance) for p in entry.get("periods", []))
    }


def mark_shared(path, surveyed, tolerance=SHARED_TOLERANCE):
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        rows = list(csv.reader(f))[1:]
    rows = [row + [""] * (len(HEADER) - len(row)) for row in rows if row]
    shared = set()
    for row in rows:
        period = float(row[2])
        others = {other[0] for other in rows if other[0] != row[0] and close(period, float(other[2]), tolerance)}
        others = sorted(others | same_sector_signals(row[0], period, surveyed), key=int)
        row[SHARED_COLUMN] = " ".join(f"TIC {tic}" for tic in others)
        if others:
            shared.add((row[0], row[1]))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return shared


def notify(title, body, urgent=False):
    env = dict(os.environ)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    command = ["notify-send", "-a", "Shadowcatch", "-u", "critical" if urgent else "normal", title, body]
    try:
        subprocess.run(command, env=env, timeout=10, check=False, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pass


def build_queue(args, tois, done):
    custom = args.ra is not None or args.dec is not None
    if custom:
        fields = [(args.ra if args.ra is not None else pick.SOUTH_POLE[0], args.dec if args.dec is not None else pick.SOUTH_POLE[1])]
    else:
        fields = list(pick.fields())
    index = core.load_index()
    used_up = index.setdefault("fields", {})
    queue = []
    for ra, dec in fields:
        key = f"{ra},{dec}"
        if not custom and not args.redo and used_up.get(key) == "used up":
            continue
        print(f"Searching TESS observations around RA {ra}, Dec {dec}...")
        counts = pick.observed_tics(ra, dec, args.box)
        fresh = {tic: n for tic, n in counts.items() if tic not in tois and str(tic) not in done and tic not in {q[1] for q in queue}}
        ranked = [r for r in pick.rank(fresh) if r[0] >= args.min_score] if fresh else []
        print(f"  {len(counts)} stars observed, {len(fresh)} not yet searched, {len(ranked)} worth searching (not giants, score >= {args.min_score:g})")
        if not custom and len(ranked) <= args.count - len(queue):
            index = core.load_index()
            index.setdefault("fields", {})[key] = "used up"
            core.save_index(index)
        queue += ranked[: args.count - len(queue)]
        if len(queue) >= args.count:
            break
    return queue


def set_paused(paused):
    core.OUTPUT.mkdir(exist_ok=True)
    if paused:
        PAUSE_FILE.touch()
        print("Survey will pause at its next step; resume with: survey.py --resume")
    else:
        PAUSE_FILE.unlink(missing_ok=True)
        print("Survey resumed")


def wait_if_paused():
    if not PAUSE_FILE.exists():
        return
    print(f"  Paused {time.strftime('%H:%M:%S')}")
    while PAUSE_FILE.exists():
        time.sleep(5)
    print(f"  Resumed {time.strftime('%H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Shadowcatch survey: search many TESS stars with no TOI and collect new candidates")
    parser.add_argument("--ra", type=float, help="search only around this point (default: work outward from the ecliptic poles)")
    parser.add_argument("--dec", type=float)
    parser.add_argument("--box", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--sectors", type=int, default=4)
    parser.add_argument("--signals", type=int, default=3)
    parser.add_argument("--min-sde", type=float, default=7.0)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="skip stars where a 2-Earth planet would be weaker than this (expected SNR)")
    parser.add_argument("--redo", action="store_true", help="search stars already surveyed again")
    parser.add_argument("--pause", action="store_true", help="pause a running survey at its next step")
    parser.add_argument("--resume", action="store_true", help="let a paused survey go on")
    args = parser.parse_args()
    if args.pause or args.resume:
        set_paused(args.pause)
        return
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)
    log_to_file()

    print(f"\n##### Shadowcatch survey started {time.strftime('%Y-%m-%d %H:%M:%S')} #####")
    print("Fetching TOI list...")
    try:
        tois = pick.known_tics()
    except Exception as e:
        traceback.print_exc()
        notify("Shadowcatch survey could not start", f"The NASA Exoplanet Archive did not answer ({type(e).__name__}). It will try again next night.", urgent=True)
        sys.exit(1)
    surveyed = core.load_index().get("surveyed", {})
    done = set() if args.redo else {tic for tic, entry in surveyed.items() if "error" not in entry}
    try:
        queue = build_queue(args, tois, done)
    except Exception as e:
        traceback.print_exc()
        notify("Shadowcatch survey could not start", f"The TESS archive did not answer ({type(e).__name__}). It will try again next night.", urgent=True)
        sys.exit(1)
    print(f"Surveying {len(queue)} stars, easiest planets to spot first\n")

    found = []
    errors = 0
    for n, (score, tic, sectors, tmag, rad) in enumerate(queue, 1):
        size = f"{rad:.2f} R_sun" if rad else "size unknown"
        print(f"=== [{n}/{len(queue)}] TIC {tic}  (Tmag {tmag}, {size}, score {score:.1f}) ===")
        start = time.time()
        try:
            result = core.run(f"TIC {tic}", args.sectors, args.signals, args.min_sde, log=lambda m: (wait_if_paused(), print("  " + m)), progress=lambda *_: wait_if_paused())
        except Exception as e:
            traceback.print_exc()
            errors += 1
            save_survey(tic, {"date": time.strftime("%Y-%m-%d"), "error": str(e)})
            continue
        new = [s for s in result["signals"] if s["label"] == "NEW CANDIDATE"]
        if new:
            core.OUTPUT.mkdir(exist_ok=True)
            append_candidates(tic, new)
            found += [(tic, s) for s in new]
        save_survey(tic, {
            "date": time.strftime("%Y-%m-%d"),
            "signals": len(result["signals"]),
            "candidates": len(new),
            "sectors": result["raw"].meta.get("SECTORS", []),
            "periods": signal_periods(result["signals"]),
        })
        print(f"  {len(new)} new candidate(s), {time.time() - start:.0f} s\n")

    shared = mark_shared(core.OUTPUT / "candidates.csv", core.load_index().get("surveyed", {}))
    systematic = [(tic, s) for tic, s in found if (str(tic), str(s["index"])) in shared]
    found = [(tic, s) for tic, s in found if (str(tic), str(s["index"])) not in shared]
    if systematic:
        print(f"{len(systematic)} candidate(s) share their period with another star's signal - likely instrumental, not counted:")
        for tic, s in systematic:
            print(f"  TIC {tic}  P={s['period']:.5f} d")
    print(f"Done {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(found)} new candidate(s) from {len(queue)} stars")
    for tic, s in found:
        print(f"  TIC {tic}  P={s['period']:.5f} d  depth={s['depth'] * 1e6:.0f} ppm  SDE={s['sde']:.1f}  -> {s['path']}")
        print(f"      CTOI values: {s['ctoi']}")
    if found:
        print(f"Full list: {core.OUTPUT / 'candidates.csv'}")
    failed = f", {errors} failed (see output/survey.log)" if errors else ""
    if found:
        lines = [f"TIC {tic}: {s['period']:.3f} d, {s['depth'] * 1e6:.0f} ppm" for tic, s in found]
        notify(
            f"Shadowcatch found {len(found)} new candidate{'s' if len(found) != 1 else ''}",
            "\n".join(lines) + f"\nSearched {len(queue)} stars{failed}. CTOI values are next to each plot in output/.",
            urgent=True,
        )
    else:
        notify("Shadowcatch survey done", f"Searched {len(queue)} stars, no new candidates{failed}.")


if __name__ == "__main__":
    main()
