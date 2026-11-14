from pathlib import Path
import pandas as pd
import json

source = Path("data/raw/xkdr/2024")
out = Path("reports/pm25_value_audit")
out.mkdir(parents=True, exist_ok=True)

files = sorted(source.glob("*.csv"))
if len(files) != 12:
    raise RuntimeError(f"Expected 12 monthly files, found {len(files)}")

frames = []
for f in files:
    df = pd.read_csv(f)
    df["_source_file"] = f.name
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

required = {"station_id", "parameter_name", "unit", "collected_at", "value"}
if not required.issubset(raw.columns):
    raise ValueError(f"Missing columns: {required - set(raw.columns)}")

if not raw["parameter_name"].astype(str).str.strip().str.upper().eq("PM2.5").all():
    raise ValueError("Unexpected pollutant found.")

raw["_numeric"] = pd.to_numeric(raw["value"], errors="coerce")
raw["_time"] = pd.to_datetime(raw["collected_at"], errors="coerce")

summaries = []
review = []

for station, g in raw.groupby("station_id"):
    v = g["_numeric"]
    t = g["_time"].dropna().drop_duplicates().sort_values()
    diffs = t.diff().dropna()

    summaries.append({
        "station_id": station,
        "rows": len(g),
        "numeric_values": int(v.notna().sum()),
        "non_numeric": int(v.isna().sum()),
        "invalid_timestamps": int(g["_time"].isna().sum()),
        "negative": int((v < 0).sum()),
        "zero": int((v == 0).sum()),
        "minus_999": int((v == -999).sum()),
        "above_500": int((v > 500).sum()),
        "above_1000": int((v > 1000).sum()),
        "min": v.min(),
        "p01": v.quantile(.01),
        "median": v.median(),
        "p95": v.quantile(.95),
        "p99": v.quantile(.99),
        "max": v.max(),
        "duplicate_station_timestamps": int(
            g.duplicated(["station_id", "collected_at"], keep=False).sum()
        ),
        "distinct_timestamps": len(t),
        "intervals_over_1h": int((diffs > pd.Timedelta(hours=1)).sum()),
        "first_timestamp": str(t.iloc[0]) if len(t) else None,
        "last_timestamp": str(t.iloc[-1]) if len(t) else None,
    })

    for _, row in g.iterrows():
        value = row["_numeric"]
        reasons = []
        if pd.isna(value):
            reasons.append("non-numeric value")
        elif value < 0:
            reasons.append("negative value")
        if value == 0:
            reasons.append("zero value")
        if pd.notna(value) and value > 500:
            reasons.append("above 500; review, not automatically invalid")
        if pd.isna(row["_time"]):
            reasons.append("invalid timestamp")

        if reasons:
            review.append({
                "station_id": station,
                "collected_at": row["collected_at"],
                "value": row["value"],
                "unit": row["unit"],
                "source_file": row["_source_file"],
                "review_reason": "; ".join(reasons)
            })

summary = pd.DataFrame(summaries)
summary.to_csv(out / "station_value_summary.csv", index=False)
pd.DataFrame(review).to_csv(out / "observations_for_review.csv", index=False)

units = raw.groupby(["station_id", "unit"], dropna=False).size()
units.rename("rows").reset_index().to_csv(out / "station_unit_summary.csv", index=False)

metadata = {
    "files_read": len(files),
    "rows_read": len(raw),
    "stations": int(raw["station_id"].nunique()),
    "raw_data_modified": False,
    "interpretation": "Flags are review prompts, not confirmed sensor faults."
}
(out / "audit_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\n=== AIRGUARD PM2.5 VALUE AUDIT ===")
print(summary.to_string(index=False))
print("\nObservations requiring review:", len(review))
print("Reports saved to:", out.resolve())
print("Raw files were not modified.")
