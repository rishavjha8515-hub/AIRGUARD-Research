# AIRGUARD
**Conservative screening of air-quality observations for possible measurement anomalies while preserving genuine pollution events.**

AIRGUARD is a research prototype, not a certified sensor-diagnostics or public-health system.

## Current research question
Can a conservative system identify station-specific measurement irregularities without incorrectly flagging genuine regional pollution episodes?

## Data status
No real dataset has been audited yet. The earlier `synthetic_airguard` report describes a deliberately corrupted test fixture only; it is not evidence about real sensors or real data feasibility.

## Candidate data sources
1. **XKDR India Air Quality Database** — hourly CPCB CAAQM and US Embassy monitor observations; published coverage Jan 2009–Mar 2026. The provider says values are published as received, without gap filling, outlier removal or calibration; unit spellings are harmonised, but values are not converted. Licence: CC BY 4.0 with attribution. See https://airquality.xkdr.org/
2. **OpenCity / CPCB Mumbai Hourly Air Quality Reports** — hourly Mumbai station reports, 2017–2023. See https://data.opencity.in/dataset/?groups=mumbai&organization=central-pollution-control-board
3. **Google Mumbai Data Challenge** — four target stations and PM2.5 forecasting data; useful as a secondary comparison source, not sufficient alone for broad spatial fault-detection claims. See https://zindi.world/competitions/google-mumbai-data-challenge/data

## First action
Run the metadata-only intake script:
```bash
python -m pip install -r requirements.txt
python scripts/inspect_xkdr.py --city Mumbai
```
It saves raw API responses under `data/raw/xkdr_metadata/`. It does not modify or clean observations.

The public demo API key is limited to 10,000 rows per query and 2024 bulk files. For broader access, obtain a free key from the XKDR site and set `XKDR_API_KEY` as an environment variable.

## Repository rules
- Keep original downloads unchanged in `data/raw/` (gitignored).
- Never silently merge station IDs, resolve conflicting duplicates, convert units, remove extremes or impute readings.
- Preserve raw timestamps and raw values.
- Record every transformation.
- Synthetic injected faults must be labelled synthetic in all reports.
- Do not claim real-world fault-detection accuracy without independently verified fault labels.
