import argparse
import math
import re
import time
import warnings
from collections import Counter
from typing import Any

warnings.filterwarnings("ignore")
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Catalogs, Observations

from core import load_index, retry, save_index

archive: Any = NasaExoplanetArchive
observations: Any = Observations
catalogs: Any = Catalogs

SOUTH_POLE = (90.0, -66.56)
NORTH_POLE = (270.0, 66.56)
MAX_STAR_RADIUS = 3.0
RING_STEP = 2.0
RINGS = 6
PATCH_MAX_AGE = 7 * 86400
VSX_MATCH = 10
GIANT_VARIABLES = {"M", "SR", "SRA", "SRB", "SRC", "SRD", "SRS", "L", "LB", "LC", "LPV", "RV", "RVA", "RVB", "ACYG", "CEP", "DCEP", "DCEPS", "CW", "CWA", "CWB"}


def known_tics():
    table = retry(lambda: archive.query_criteria(table="toi", select="tid"))
    return {int(t) for t in table["tid"]}


def observed_tics(ra, dec, box):
    key = f"{ra},{dec},{box}"
    index = load_index()
    cached = index.setdefault("patches", {}).get(key)
    if cached and time.time() - cached["time"] < PATCH_MAX_AGE:
        return Counter({int(tic): n for tic, n in cached["counts"].items()})
    width = box / max(math.cos(math.radians(dec)), 0.1)
    obs = retry(lambda: observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        provenance_name="SPOC",
        s_ra=[ra - width, ra + width],
        s_dec=[dec - box, dec + box],
    ))
    counts = Counter(int(name) for name in obs["target_name"] if str(name).isdigit())
    index = load_index()
    index.setdefault("patches", {})[key] = {"time": time.time(), "counts": {str(tic): n for tic, n in counts.items()}}
    save_index(index)
    return counts


def star_params(tics):
    params = {}
    tics = list(tics)
    for i in range(0, len(tics), 500):
        table = retry(lambda: catalogs.query_criteria(catalog="Tic", ID=tics[i : i + 500]))
        for row in table:
            tmag = None if row["Tmag"] is None or math.isnan(row["Tmag"]) else float(row["Tmag"])
            rad = None if row["rad"] is None or math.isnan(row["rad"]) else float(row["rad"])
            params[int(row["ID"])] = (tmag, rad, str(row["lumclass"]) == "GIANT", float(row["ra"]), float(row["dec"]))
    return params


def giant_variables(stars):
    index = load_index()
    known = index.setdefault("vsx_giants", {})
    todo = [(tic, ra, dec) for tic, ra, dec in stars if str(tic) not in known]
    for i in range(0, len(todo), 500):
        chunk = todo[i : i + 500]
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from astroquery.vizier import Vizier

            vizier: Any = Vizier(columns=["Type"], row_limit=-1)
            center = SkyCoord([ra for _, ra, _ in chunk], [dec for _, _, dec in chunk], unit="deg")
            tables = vizier.query_region(center, radius=VSX_MATCH * u.arcsec, catalog="B/vsx/vsx")
        except Exception as e:
            print(f"Could not check VSX for giant variables: {e}")
            break
        found = set()
        if tables:
            for row in tables[0]:
                if set(re.split(r"[/+:|()]", str(row["Type"]))) & GIANT_VARIABLES:
                    found.add(int(row["_q"]) - 1)
        for k, (tic, _, _) in enumerate(chunk):
            known[str(tic)] = k in found
        save_index(index)
    return {tic for tic, _, _ in stars if known.get(str(tic))}


def detectability(tmag, rad, sectors):
    depth = (2 / 109.1 / (rad or 1.0)) ** 2 * 1e6
    noise = max(60 * 10 ** (0.2 * ((tmag or 12) - 9)), 30)
    return depth / noise * math.sqrt(sectors)


def rank(counts):
    params = star_params(counts)
    unknown = [(tic, p[3], p[4]) for tic, p in params.items() if tic in counts and p[1] is None and not p[2]]
    variables = giant_variables(unknown)
    ranked = []
    for tic, sectors in counts.items():
        tmag, rad, giant, *_ = params.get(tic, (None, None, False))
        if giant or tic in variables or (rad is not None and rad > MAX_STAR_RADIUS):
            continue
        ranked.append((detectability(tmag, rad, sectors), tic, sectors, tmag, rad))
    return sorted(ranked, reverse=True)


def fields():
    for pole in (SOUTH_POLE, NORTH_POLE):
        for ring in range(RINGS + 1):
            radius = ring * RING_STEP
            count = max(1, round(2 * math.pi * radius / RING_STEP))
            for k in range(count):
                yield offset(pole, radius, 360 * k / count)


def offset(center, radius, angle):
    ra0, dec0 = map(math.radians, center)
    r, a = math.radians(radius), math.radians(angle)
    dec = math.asin(math.sin(dec0) * math.cos(r) + math.cos(dec0) * math.sin(r) * math.cos(a))
    ra = ra0 + math.atan2(math.sin(a) * math.sin(r) * math.cos(dec0), math.cos(r) - math.sin(dec0) * math.sin(dec))
    return round(math.degrees(ra) % 360, 3), round(math.degrees(dec), 3)


def main():
    parser = argparse.ArgumentParser(description="Find well-observed TESS stars with no TOI, easiest planets first")
    parser.add_argument("--ra", type=float, default=SOUTH_POLE[0])
    parser.add_argument("--dec", type=float, default=SOUTH_POLE[1])
    parser.add_argument("--box", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    print("Fetching TOI list...")
    tois = known_tics()
    print(f"Searching TESS observations around RA {args.ra}, Dec {args.dec}...")
    counts = observed_tics(args.ra, args.dec, args.box)
    fresh = Counter({tic: n for tic, n in counts.items() if tic not in tois})
    ranked = rank(fresh)
    print(f"{len(counts)} stars observed, {len(fresh)} without a TOI, {len(ranked)} not giants\n")
    for score, tic, n, tmag, rad in ranked[: args.count]:
        size = f"{rad:.2f} R_sun" if rad else "size unknown"
        print(f"TIC {tic}   {n} sectors   Tmag {tmag}   {size}   score {score:.1f}")


if __name__ == "__main__":
    main()
