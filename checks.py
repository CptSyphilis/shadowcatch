import sys
import time

import core

PLANET = "planet"
NOT_NEW = "not new"
BINARY = "binary"
PIXELS = "pixels"
PIXELS_ALL = "pixels across sectors"
RECHECK = "other sectors"

CASES = [
    ("Pi Mensae", 3, [(6.268, PLANET, "known TOI 144.01"), (6.268, PIXELS, "saturated"), (6.268, RECHECK, "seen again"), (None, NOT_NEW, "")]),
    ("TOI-270", 3, [(5.660, PLANET, "known TOI 270.01"), (3.360, PLANET, "known TOI 270.03"), (5.660, PIXELS, "on target")]),
    ("TIC 279740441", 3, [(0.7419, PIXELS, "off target")]),
    ("TIC 81212286", 3, [(0.6851, PIXELS, "too close to tell")]),
    ("TOI-216", 3, [(34.446, PIXELS_ALL, "on target"), (34.446, RECHECK, "not missing")]),
    ("WASP-18", 2, [(0.9415, PLANET, "known TOI 185.01")]),
    ("WASP-121", 2, [(1.2749, PLANET, "known TOI 495.01")]),
    ("LHS 3844", 3, [(0.4629, PLANET, "known TOI 136.01")]),
    ("TIC 22767596", 2, [(2.5570, PLANET, "known CTOI 22767596.01")]),
    ("TIC 41232835", 3, [(None, BINARY, "binary")]),
    ("TIC 41232189", 2, [(None, NOT_NEW, "")]),
    ("TIC 41168549", 4, [(None, NOT_NEW, "")]),
    ("TIC 41230564", 4, [(None, NOT_NEW, "")]),
    ("TIC 389865785", 4, [(None, NOT_NEW, "")]),
    ("TIC 31514067", 4, [(None, NOT_NEW, "")]),
    ("TIC 41335086", 4, [(None, NOT_NEW, "")]),
    ("TIC 374942490", 4, [(None, NOT_NEW, "")]),
    ("TIC 149992221", 4, [(6.990, NOT_NEW, "not confirmed")]),
    ("TIC 374861595", 4, [(None, BINARY, "too big")]),
]


def find(signals, period):
    for s in signals:
        if abs(s["period"] - period) / period < 0.01:
            return s
    return None


def check(target, sectors, expectations):
    result = core.run(target, sectors, 3, 7.0, log=core.quiet)
    signals = result["signals"]
    problems = []
    for period, kind, text in expectations:
        if kind == PLANET:
            s = find(signals, period)
            if s is None:
                problems.append(f"no signal near {period} d (got {[round(x['period'], 4) for x in signals]})")
            elif s["label"] != text:
                problems.append(f"{period} d labeled '{s['label']}', expected '{text}'")
            elif s["flags"]:
                problems.append(f"{period} d is a real planet but got flags {[f[0] for f in s['flags']]}")
        elif kind == BINARY:
            if not signals or text not in signals[0]["label"]:
                problems.append(f"first signal should be a binary, got {[s['label'] for s in signals]}")
        elif kind in (PIXELS, PIXELS_ALL):
            s = find(signals, period)
            if s is None:
                problems.append(f"no signal near {period} d for the pixel check")
            else:
                neighbors = core.gaia_neighbors(result["star"], log=core.quiet)
                s["blends"] = core.blend_check(s, neighbors) if neighbors else None
                pixels = core.pixel_check(s, result["star"], log=core.quiet, max_sectors=1 if kind == PIXELS else 4, neighbors=neighbors)
                verdict = pixels["verdict"] if pixels else None
                if verdict != text:
                    problems.append(f"pixel check at {period} d says '{verdict}', expected '{text}'")
        elif kind == RECHECK:
            s = find(signals, period)
            check = None if s is None else core.recheck(s, result["star"], log=core.quiet)
            verdict = check["verdict"] if check else None
            if (verdict == text[4:]) if text.startswith("not ") else (verdict != text):
                problems.append(f"other-sector check at {period} d says '{verdict}', expected '{text}'")
        elif kind == NOT_NEW:
            new = [round(s["period"], 4) for s in signals if s["label"] == "NEW CANDIDATE"]
            if new:
                problems.append(f"false candidates at {new} d")
            if period is not None:
                s = find(signals, period)
                if s is None or text not in s["label"]:
                    problems.append(f"{period} d should be labeled '{text}...', got {s['label'] if s else 'no signal'}")
    return problems, [f"{s['period']:.4f} d {s['label']}" for s in signals]


def main():
    names = sys.argv[1:]
    cases = [c for c in CASES if not names or c[0] in names]
    failed = 0
    for target, sectors, expectations in cases:
        start = time.time()
        try:
            problems, found = check(target, sectors, expectations)
        except Exception as e:
            problems, found = [f"crashed: {type(e).__name__}: {e}"], []
        verdict = "PASS" if not problems else "FAIL"
        failed += bool(problems)
        print(f"{verdict}  {target}  ({time.time() - start:.0f} s)  {'; '.join(found)}", flush=True)
        for problem in problems:
            print(f"      {problem}", flush=True)
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
