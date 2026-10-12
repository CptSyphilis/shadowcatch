# Shadowcatch

A planet hunter for NASA's TESS space telescope. It downloads a star's brightness measurements, searches them for small repeating dips (a planet passing in front of its star), and checks each dip against the catalogs and a set of false-alarm tests. A dip that survives all of them is flagged as a **new candidate**, meaning one that isn't already listed as a TESS Object of Interest (TOI) or a community TOI (CTOI).

It runs as a command-line tool, a desktop app, or an unattended survey that works through hundreds of stars on its own.

## Install

Needs Python 3.10+ and an internet connection (data comes from MAST, the NASA Exoplanet Archive, ExoFOP, VizieR and SIMBAD).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The GUI also needs Tk (`sudo apt install python3-tk` on Debian/Ubuntu). Desktop notifications from the survey use `notify-send` (Linux).

## Usage

### One star

```
.venv/bin/python core.py "Pi Mensae"
.venv/bin/python core.py "TIC 261136679" --sectors 4
```

| Option | Default | Meaning |
|---|---|---|
| `--sectors N` | 10 | how many TESS sectors (~27-day observing runs) to use |
| `--signals N` | 3 | how many separate dip patterns to look for |
| `--min-sde X` | 7.0 | how strongly a signal must stand out from noise to be reported |

3–6 sectors is the sweet spot. The search time grows with the square of the time span.

### Desktop app

```
.venv/bin/python gui.py
```

Type a star name or TIC ID and press search. Each signal found shows up as a card (gold = new candidate, cyan = known planet, grey = not a planet). Click a card to see its plots and a plain-language explanation. The **Help** button has a beginner's guide.

### Survey many stars

```
.venv/bin/python survey.py                 # 10 stars
.venv/bin/python survey.py --count 20      # 20 stars
```

Works outward from the two TESS continuous viewing zones (the patches of sky TESS watches longest). It picks the stars with no known planets where a small planet would be easiest to spot, and skips any star it has already searched. New candidates are appended to `output/candidates.csv`, and a desktop notification is sent when the survey finishes.

Useful options: `--sectors` (default 4), `--min-score` (skip stars where a 2-Earth-radius planet would be too faint), `--ra/--dec` (survey one patch of sky), `--redo` (search stars again).

To run it every night, add a cron job (`crontab -e`):

```
0 2 * * * cd /path/to/shadowcatch && .venv/bin/python survey.py --count 20 >> output/survey.log 2>&1
```

### Find targets by hand

```
.venv/bin/python pick.py --ra 90 --dec -66.5 --count 10
```

Lists the best stars without a known planet in a patch of sky.

## How it works

1. **Download.** Gets light curves (brightness over time) from MAST, preferring 2-minute SPOC data and nearby sectors. Files already on disk are reused.
2. **Clean.** Cuts the edges of data gaps, removes flares, flattens slow brightness changes, and bins to 10 minutes.
3. **Search.** Runs a Box Least Squares periodogram from 0.2 days up to half the time span. Each signal found is masked so the next search can find another.
4. **Vet.** Rejects dips that don't behave like a planet:
   - too big for a planet
   - alternating depths or a second eclipse at half-phase (two stars)
   - V-shaped instead of flat-bottomed
   - smooth waves (spots, pulsations)
   - depth that changes over time
   - an orbit that would fit inside the star
   - a dip too short or too long for that star's density
   - one single dip carrying most of the signal
   - too few dips, or too weak against the star's own flickering
5. **Compare with what's known.** Checks the TOI list, ExoFOP's community candidates and ExoFOP follow-up notes, plus the VSX and SIMBAD variable-star and binary catalogs.
6. **Re-check.** Looks for the same dip in every other TESS sector of that star.
7. **Neighbors.** Gaia lists nearby stars that could fake the dip if they were eclipsing binaries. A pixel-level difference image then shows whether the light is lost on the target or on a neighbor.
8. **Report.** Saves a plot per signal and, for new candidates, a file of values laid out like ExoFOP's CTOI table.

## Output

```
output/
  candidates.csv           every new candidate from the survey, with all check results
  survey.log               survey history
  <star>/signal<N>.png     light curve, periodogram, folded dip (and pixel images for candidates)
  <star>/signal<N>_ctoi.txt  CTOI-style values for a new candidate
```

`cache.json` remembers star lookups, catalog queries and which stars have been surveyed. Light curves are cached by lightkurve in `~/.cache/lightkurve`.

### Reading a label

| Label | Meaning |
|---|---|
| `NEW CANDIDATE` | passed every check and isn't in any catalog; worth a closer look |
| `known TOI X` / `known CTOI X` | someone already found it (a good sign the pipeline works) |
| `alias of ...` / `repeat of signal N` | the same planet seen at double, half or third its period |
| `likely noise (...)` | too few dips, too weak, too short, or one dip only |
| `likely variable star (...)` | the star's own spots or pulsations |
| `likely binary (...)` / `catalogued binary` | two stars eclipsing each other |
| `not confirmed (...)` | the dip doesn't show up in the star's other TESS data |
| `likely blend (dip is off target)` | the dip comes from a neighboring star |

A new candidate is **not** a discovery. Most turn out to be instrumental effects or blended binaries. A candidate whose period matches other stars' candidates (see `shared_period_with` in the CSV) is almost always instrumental, since a TESS sector is ~27 days long.

## Regression checks

```
.venv/bin/python checks.py              # all cases (~5 min with data cached)
.venv/bin/python checks.py "Pi Mensae"  # one case
```

Runs known planets (Pi Men c, TOI-270, WASP-18 b, WASP-121 b, LHS 3844 b, TOI-216 c) and stars that once produced false candidates through the pipeline, and checks every label. Run it after changing any threshold.

## Reporting a candidate

ExoFOP only accepts community candidates (CTOIs) that have already been published in a peer-reviewed journal, uploaded by approved users. To get an unpublished candidate followed up, share the values in its `_ctoi.txt` file with a researcher.

## Data and credits

TESS data from the Mikulski Archive for Space Telescopes (MAST); TOI list from the NASA Exoplanet Archive; community candidates from ExoFOP; Gaia DR3 and VSX via VizieR; SIMBAD (CDS, Strasbourg). Built on [lightkurve](https://docs.lightkurve.org), [astropy](https://www.astropy.org) and [astroquery](https://astroquery.readthedocs.io).

## License

MIT, see [LICENSE](LICENSE).
