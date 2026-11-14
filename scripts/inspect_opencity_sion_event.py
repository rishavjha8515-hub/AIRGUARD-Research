import pandas as pd

path = "data/raw/opencity/sion_mpcb_2024_25.csv"
df = pd.read_csv(path)

# Preserve source timestamps as UTC, then convert explicitly to IST.
df["Timestamp_UTC"] = pd.to_datetime(df["Timestamp"], utc=True)
df["Timestamp_IST"] = df["Timestamp_UTC"].dt.tz_convert("Asia/Kolkata")

start = pd.Timestamp("2024-07-06 00:00:00", tz="Asia/Kolkata")
end = pd.Timestamp("2024-07-08 00:00:00", tz="Asia/Kolkata")

window = df[df["Timestamp_IST"].between(start, end)].copy()

print("=== SOURCE CHECK ===")
print("Station names:", df["Station Name"].dropna().unique())
print("Rows in event window:", len(window))
print("UTC range:", df["Timestamp_UTC"].min(), "to", df["Timestamp_UTC"].max())

print("\n=== 15-MINUTE OBSERVATIONS (IST) ===")
cols = [
    "Timestamp_IST", "PM2.5 (µg/m³)", "PM10 (µg/m³)",
    "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)",
    "RF (mm)", "TOT-RF (mm)"
]
print(window[cols].to_string(index=False))

print("\n=== HOURLY SUMMARY (IST) ===")
hourly = window.set_index("Timestamp_IST").resample("1h").agg({
    "PM2.5 (µg/m³)": ["count", "mean", "median", "max"],
    "PM10 (µg/m³)": "mean",
    "AT (°C)": "mean",
    "RH (%)": "mean",
    "WS (m/s)": "mean",
    "WD (deg)": "mean",
    "RF (mm)": "sum"
})
print(hourly.to_string())
