from pathlib import Path
import pandas as pd

path = Path("reports/temporal_baseline/temporal_scores.csv")
df = pd.read_csv(path)

for col in [
    "value", "baseline_median", "baseline_mad",
    "baseline_count", "robust_z"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

original = (
    df["candidate_anomaly"]
    .astype(str).str.lower().eq("true")
)

eligible = (
    df["value"].notna()
    & df["baseline_median"].notna()
    & df["baseline_mad"].notna()
    & (df["baseline_count"] >= 12)
)

deviation = (df["value"] - df["baseline_median"]).abs()
raw_scale = 1.4826 * df["baseline_mad"]

original_recomputed = (
    eligible
    & (raw_scale > 0)
    & ((deviation / raw_scale) >= 6)
)

floor_score = deviation / raw_scale.clip(lower=1.0)
floor_flags = eligible & (floor_score >= 6)

print("=== ELIGIBILITY ===")
print("Original candidate flags:", int(original.sum()))
print("Original robust_z non-null:", int(df["robust_z"].notna().sum()))
print("Recomputed eligible:", int(eligible.sum()))

print("\n=== ORIGINAL IMPLEMENTATION CHECK ===")
print("Stored vs recomputed disagreements:",
      int((original != original_recomputed).sum()))

print("\n=== VARIANT CHECK ===")
print("Original flags:", int(original.sum()))
print("Recomputed original flags:", int(original_recomputed.sum()))
print("Scale-floor flags:", int(floor_flags.sum()))

print("\n=== B-ONLY OBSERVATIONS ===")
b_only = df[floor_flags & ~original].copy()
print("Count:", len(b_only))
print(b_only[[
    "station_id", "collected_at", "value",
    "baseline_median", "baseline_mad",
    "baseline_count", "robust_z"
]].head(30).to_string(index=False))

print("\n=== A-ONLY OBSERVATIONS ===")
a_only = df[original & ~floor_flags].copy()
print("Count:", len(a_only))
print(a_only[[
    "station_id", "collected_at", "value",
    "baseline_median", "baseline_mad",
    "baseline_count", "robust_z"
]].head(30).to_string(index=False))
zero_mad = eligible & (df["baseline_mad"] == 0)

print("\n=== ZERO-MAD AUDIT ===")
print("Zero-MAD observations:", int(zero_mad.sum()))
print("Zero-MAD B candidates:", int((zero_mad & floor_flags).sum()))

print("\nBy station:")
print(
    df[zero_mad]
    .groupby("station_id")
    .agg(
        observations=("value", "size"),
        distinct_baselines=("baseline_median", "nunique"),
        distinct_values=("value", "nunique"),
    )
    .to_string()
)

print("\nZero-MAD candidate examples:")
print(
    df[zero_mad & floor_flags][[
        "station_id", "collected_at", "value",
        "baseline_median", "baseline_count"
    ]].head(40).to_string(index=False)
)