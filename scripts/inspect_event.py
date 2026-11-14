import pandas as pd
from pathlib import Path

files = sorted(Path("data/raw/xkdr/2024").glob("*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"])

start = pd.Timestamp("2024-07-07 06:00:00")
end   = pd.Timestamp("2024-07-08 06:00:00")

event = df[
    df["collected_at"].between(start, end)
].copy()

event = event.sort_values(["collected_at", "station_id"])

print("=== ALL STATIONS: JULY 7–8 EVENT WINDOW ===")
print(event.pivot(
    index="collected_at",
    columns="station_id",
    values="value"
).to_string())

print("\n=== SION: LOCAL EVENT ===")
print(
    event[event["station_id"] == "site_5119"]
    [["collected_at", "value"]].to_string(index=False)
)
