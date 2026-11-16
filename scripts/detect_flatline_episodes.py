
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/xkdr/2024")
OUT_DIR = Path("reports/airguard_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_HOURS = 12

files = sorted(DATA_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No monthly CSV files found in {DATA_DIR}")

df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["station_id", "collected_at", "value"])
df = df.sort_values(["station_id", "collected_at"])

episodes = []

for station_id, group in df.groupby("station_id"):
    group = group.reset_index(drop=True)

    run_start = 0

    for i in range(1, len(group) + 1):
        continues = False

        if i < len(group):
            same_value = group.loc[i, "value"] == group.loc[i - 1, "value"]
            hourly_gap = (
                group.loc[i, "collected_at"] -
                group.loc[i - 1, "collected_at"]
            ) == pd.Timedelta(hours=1)

            continues = same_value and hourly_gap

        if not continues:
            run = group.iloc[run_start:i]
            duration = len(run)

            if duration >= MIN_HOURS:
                episodes.append({
                    "station_id": station_id,
                    "start": run.iloc[0]["collected_at"],
                    "end": run.iloc[-1]["collected_at"],
                    "duration_hours": duration,
                    "value": run.iloc[0]["value"],
                })

            run_start = i

result = pd.DataFrame(episodes)

if not result.empty:
    result = result.sort_values(
        ["duration_hours", "station_id"],
        ascending=[False, True]
    )

output = OUT_DIR / "flatline_episodes.csv"
result.to_csv(output, index=False)

print("=== FLATLINE EPISODES ===")
print(f"Minimum consecutive hours: {MIN_HOURS}")
print(f"Episodes found: {len(result)}")

if not result.empty:
    print(result.to_string(index=False))

print(f"\nSaved: {output}")
