
from pathlib import Path
import pandas as pd

path = Path("data/raw/opencity/sion_mpcb_2024_25.csv")
df = pd.read_csv(path)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"].astype(str).str.replace(r"\+0000$", "", regex=True),
    errors="coerce"
)

start = pd.Timestamp("2024-09-19 18:00:00")
end = pd.Timestamp("2024-09-23 00:00:00")

event = df.loc[
    (df["Station ID"] == "site_5119")
    & (df["Timestamp"] >= start)
    & (df["Timestamp"] <= end),
    ["Timestamp", "PM2.5 (µg/m³)"]
].copy()

event = event.sort_values("Timestamp")
event["PM2.5 (µg/m³)"] = pd.to_numeric(
    event["PM2.5 (µg/m³)"], errors="coerce"
)

print("Rows:", len(event))
print("Distinct PM2.5 values:", event["PM2.5 (µg/m³)"].nunique())
print("Timestamp duplicates:", event["Timestamp"].duplicated().sum())

print("\n=== HOURLY SOURCE RECORDS ===")
print(
    event.groupby(event["Timestamp"].dt.floor("h"))
    ["PM2.5 (µg/m³)"]
    .agg(["count", "nunique", "min", "mean", "max"])
    .to_string()
)

print("\n=== FIRST AND LAST RECORDS ===")
print(event.head(12).to_string(index=False))
print(event.tail(12).to_string(index=False))

print("\n=== RECORDS DURING FLATLINE ===")
flat = event.loc[
    (event["Timestamp"] >= pd.Timestamp("2024-09-20 00:00:00"))
    & (event["Timestamp"] <= pd.Timestamp("2024-09-22 21:00:00"))
]
print(flat.to_string(index=False))
