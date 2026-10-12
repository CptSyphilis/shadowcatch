import argparse
import math
import warnings
from collections import Counter
from typing import Any

warnings.filterwarnings("ignore")
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Catalogs, Observations

archive: Any = NasaExoplanetArchive
observations: Any = Observations
catalogs: Any = Catalogs

SOUTH_POLE = (90.0, -66.56)
NORTH_POLE = (270.0, 66.56)
MAX_STAR_RADIUS = 3.0
RING_STEP = 2.0
RINGS = 6


def known_tics():
    table = archive.query_criteria(table="toi", select="tid")
    return {int(t) for t in table["tid"]}


def observed_tics(ra, dec, box):
    width = box / max(math.cos(math.radians(dec)), 0.1)
    obs = observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        provenance_name="SPOC",
        s_ra=[ra - width, ra + width],
        s_dec=[dec - box, dec + box],
    )
    return Counter(int(name) for name in obs["target_name"] if str(name).isdigit())


def star_params(tics):
    params = {}
    tics = list(tics)
    for i in range(0, len(tics), 500):
        table = catalogs.query_criteria(catalog="Tic", ID=tics[i : i + 500])
        for row in table:
            tmag = None if row["Tmag"] is None or math.isnan(row["Tmag"]) else float(row["Tmag"])
            rad = None if row["rad"] is None or math.isnan(row["rad"]) else float(row["rad"])
            params[int(row["ID"])] = (tmag, rad, str(row["lumclass"]) == "GIANT")
    return params


def detectability(tmag, rad, sectors):
    depth = (2 / 109.1 / (rad or 1.0)) ** 2 * 1e6
    noise = max(60 * 10 ** (0.2 * ((tmag or 12) - 9)), 30)
    return depth / noise * math.sqrt(sectors)


def rank(counts):
    params = star_params(counts)
    ranked = []
    for tic, sectors in counts.items():
        tmag, rad, giant = params.get(tic, (None, None, False))
        if giant or (rad is not None and rad > MAX_STAR_RADIUS):
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
