from pathlib import Path
import pandas as pd

files = sorted(Path("data/raw/xkdr/2024").glob("*.csv"))
df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.sort_values(["station_id", "collected_at"])

episodes = [
    {
        "station": "site_5104",
        "start": "2024-10-21 00:00:00",
        "end": "2024-10-25 01:00:00",
        "label": "Kurla October episode",
    },
    {
        "station": "site_5119",
        "start": "2024-09-19 13:00:00",
        "end": "2024-09-22 21:00:00",
        "label": "Sion September episode",
    },
]

for episode in episodes:
    x = df[
        (df["station_id"] == episode["station"])
        & (df["collected_at"] >= episode["start"])
        & (df["collected_at"] <= episode["end"])
    ].copy()

    x = x.sort_values("collected_at")
    x["time_gap"] = x["collected_at"].diff()

    # A new run begins if the value changes or the hourly sequence breaks.
    x["new_run"] = (
        x["value"].ne(x["value"].shift())
        | x["time_gap"].ne(pd.Timedelta(hours=1))
    )
    x["run_id"] = x["new_run"].cumsum()

    runs = (
        x.groupby("run_id")
        .agg(
            start=("collected_at", "min"),
            end=("collected_at", "max"),
            readings=("value", "size"),
            value=("value", "first"),
            distinct_values=("value", "nunique"),
        )
        .reset_index(drop=True)
    )

    print(f"\n=== {episode['label']} ===")
    print("Rows:", len(x))
    print("Distinct readings:", x["value"].nunique())
    print("Missing hourly intervals:", int(
        (x["collected_at"].diff() > pd.Timedelta(hours=1)).sum()
    ))
    print("Exact repeated-value runs (3+ consecutive readings):")

    repeated = runs[runs["readings"] >= 3].sort_values(
        "readings", ascending=False
    )
    if repeated.empty:
        print("None")
    else:
        print(repeated.to_string(index=False))

    print("\nRaw hourly sequence:")
    print(x[["collected_at", "value"]].to_string(index=False))