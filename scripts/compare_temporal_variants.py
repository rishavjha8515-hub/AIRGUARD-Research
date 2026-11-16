
from pathlib import Path
import pandas as pd

INPUT = Path("reports/temporal_baseline/temporal_scores.csv")
OUT = Path("reports/temporal_baseline")

THRESHOLD = 6.0
MIN_SCALE = 1.0       # µg/m³, experimental minimum robust scale
MIN_ABS_DEVIATION = 25.0  # µg/m³, exploratory sensitivity test

df = pd.read_csv(INPUT)

required = {
    "station_id", "collected_at", "value",
    "baseline_median", "baseline_mad",
    "baseline_count", "robust_z", "candidate_anomaly",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for col in [
    "value", "baseline_median", "baseline_mad",
    "baseline_count", "robust_z",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Variant A: original detector.
df["A_original"] = df["candidate_anomaly"].astype(str).str.lower().eq("true")

# A baseline is eligible when it has the required history.
eligible = (
    df["baseline_median"].notna()
    & df["baseline_mad"].notna()
    & (df["baseline_count"] >= 12)
)

deviation = (df["value"] - df["baseline_median"]).abs()
raw_scale = 1.4826 * df["baseline_mad"]

# Variant B: same threshold, but a minimum robust scale of 1 µg/m³.
floor_scale = raw_scale.clip(lower=MIN_SCALE)
df["B_floor_score"] = deviation / floor_scale
df["B_scale_floor"] = eligible & (df["B_floor_score"] >= THRESHOLD)

# Variant C: original flag plus a minimum absolute deviation.
df["C_absolute_filter"] = (
    df["A_original"] & (deviation >= MIN_ABS_DEVIATION)
)

variants = {
    "A_original": df["A_original"],
    "B_scale_floor": df["B_scale_floor"],
    "C_absolute_filter": df["C_absolute_filter"],
}

# Overall comparison.
rows = []
for name, flags in variants.items():
    rows.append({
        "variant": name,
        "eligible_observations": int(
            eligible.sum() if name != "A_original"
            else df["robust_z"].notna().sum()
        ),
        "candidate_count": int(flags.sum()),
        "candidate_rate_percent": round(
            100 * flags.sum() / max(1, eligible.sum()), 4
        ),
        "overlap_with_original": int(
            (flags & df["A_original"]).sum()
        ),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "variant_comparison_summary.csv", index=False)

# Station-level comparison.
station_rows = []
for station, group in df.groupby("station_id"):
    for name, _ in variants.items():
        flags = group[name]
        station_rows.append({
            "station_id": station,
            "variant": name,
            "candidate_count": int(flags.sum()),
        })

station_summary = pd.DataFrame(station_rows)
station_summary.to_csv(
    OUT / "variant_station_summary.csv", index=False
)

# Preserve the known Sion episode for direct comparison.
df["collected_at"] = pd.to_datetime(df["collected_at"])
sion = df[
    (df["station_id"] == "site_5119")
    & (df["collected_at"] >= "2024-07-06")
    & (df["collected_at"] < "2024-07-09")
].copy()

sion_columns = [
    "collected_at", "value", "baseline_median",
    "baseline_mad", "baseline_count", "robust_z",
    "A_original", "B_floor_score",
    "B_scale_floor", "C_absolute_filter",
]
sion[sion_columns].to_csv(
    OUT / "sion_variant_comparison.csv", index=False
)

print("=== OVERALL VARIANT COMPARISON ===")
print(summary.to_string(index=False))

print("\n=== STATION-LEVEL COUNTS ===")
print(station_summary.pivot(
    index="station_id",
    columns="variant",
    values="candidate_count",
).to_string())

print("\n=== SION JULY 6–8 ===")
print(sion[sion_columns].to_string(index=False))

print("\nSaved comparison reports to:", OUT.resolve())
print("Original scores and raw observations were not modified.")
