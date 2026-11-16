
from pathlib import Path
import pandas as pd

INPUT = Path("reports/temporal_baseline/temporal_scores.csv")
OUTPUT = Path("reports/temporal_baseline")

df = pd.read_csv(INPUT)
df["collected_at"] = pd.to_datetime(df["collected_at"])

scored = df.dropna(subset=["robust_z", "baseline_mad"]).copy()

print("=== BASELINE SCALE AUDIT ===")
print("Scored observations:", len(scored))
print("Zero MAD:", (scored["baseline_mad"] == 0).sum())
print("MAD below 0.5:", (scored["baseline_mad"] < 0.5).sum())
print("MAD below 1.0:", (scored["baseline_mad"] < 1.0).sum())
print("MAD below 2.0:", (scored["baseline_mad"] < 2.0).sum())

print("\n=== STATION-LEVEL SUMMARY ===")

summary = scored.groupby("station_id").agg(
    observations=("value", "size"),
    median_mad=("baseline_mad", "median"),
    p10_mad=("baseline_mad", lambda x: x.quantile(0.10)),
    fraction_mad_below_1=("baseline_mad", lambda x: (x < 1).mean()),
    candidate_count=("candidate_anomaly", "sum"),
    candidate_rate=("candidate_anomaly", "mean"),
)

summary["candidate_rate"] *= 100
summary["fraction_mad_below_1"] *= 100

print(summary.round(2).to_string())

print("\n=== CANDIDATE SCORE DISTRIBUTION ===")
print(
    scored.loc[scored["candidate_anomaly"], "robust_z"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .round(2)
    .to_string()
)

summary.to_csv(OUTPUT / "baseline_scale_summary.csv")

print("\nSaved baseline_scale_summary.csv")

