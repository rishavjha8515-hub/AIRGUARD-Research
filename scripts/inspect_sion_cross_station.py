import pandas as pd
from pathlib import Path

files = sorted(Path("data/raw/xkdr/2024").glob("*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"])

start = "2024-07-06 18:00:00"
end = "2024-07-07 17:00:00"

window = df[df["collected_at"].between(start, end)]

pivot = window.pivot_table(
    index="collected_at",
    columns="station_id",
    values="value",
    aggfunc="first"
).sort_index()

print(pivot.to_string())
