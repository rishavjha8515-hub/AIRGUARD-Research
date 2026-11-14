import pandas as pd
from pathlib import Path

files = sorted(Path("data/raw/xkdr/2024").glob("*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"])

start = "2024-07-05 00:00:00"
end = "2024-07-11 00:00:00"

window = df[
    (df["station_id"] == "site_5119") &
    df["collected_at"].between(start, end)
].sort_values("collected_at")

print("=== SION: JULY 5–10 ===")
print(window[["collected_at", "value"]].to_string(index=False))

print("\n=== DAILY SUMMARY ===")
print(window.groupby(window["collected_at"].dt.date)["value"].agg(
    ["count", "min", "median", "max"]
).to_string())
