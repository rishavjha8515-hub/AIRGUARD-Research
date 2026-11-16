
import pandas as pd
import numpy as np

old_path = "reports/airguard_ml/test_predictions.csv"
new_path = "reports/airguard_ml/station_model_comparison.csv"

old = pd.read_csv(old_path, parse_dates=["collected_at"])
new = pd.read_csv(new_path, parse_dates=["collected_at"])

keys = ["station_id", "collected_at"]

features = [
    "value",
    "past_median",
    "past_mad",
    "robust_deviation",
    "absolute_change",
    "change_rate",
    "repeated_run",
]

merged = old.merge(
    new,
    on=keys,
    how="inner",
    suffixes=("_old", "_new"),
    validate="one_to_one"
)

print("=== REPRODUCIBILITY CHECK ===")
print("Original rows:", len(old))
print("New rows:", len(new))
print("Matched rows:", len(merged))

print("\n=== FEATURE COMPARISON ===")

for feature in features:
    a = merged[f"{feature}_old"]
    b = merged[f"{feature}_new"]

    equal = np.isclose(
        a,
        b,
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True
    )

    print(f"{feature}: {int(equal.sum())}/{len(merged)} match")

    if not equal.all():
        difference = (a - b).abs()

        print("  Maximum difference:", difference.max())

        examples = merged.loc[
            ~equal,
            ["station_id", "collected_at",
             f"{feature}_old", f"{feature}_new"]
        ].head(5)

        print("  First differences:")
        print(examples.to_string(index=False))

print("\n=== GLOBAL FLAG COMPARISON ===")

old_flags = merged["ml_candidate"].astype(bool)
new_flags = merged["global_ml_flag"].astype(bool)

print("Original flags:", int(old_flags.sum()))
print("New global flags:", int(new_flags.sum()))
print("Both flag:", int((old_flags & new_flags).sum()))
print("Old only:", int((old_flags & ~new_flags).sum()))
print("New only:", int((~old_flags & new_flags).sum()))

print("\n=== SCORE COMPARISON ===")

difference = (
    merged["anomaly_score"]
    - merged["global_anomaly_score"]
).abs()

print("Maximum absolute score difference:", difference.max())
print("Mean absolute score difference:", difference.mean())

print("\n=== FLAG DIFFERENCES BY STATION ===")

merged["old_only"] = old_flags & ~new_flags
merged["new_only"] = ~old_flags & new_flags

print(
    merged.groupby("station_id")
    .agg(
        old_flags=("ml_candidate", "sum"),
        new_flags=("global_ml_flag", "sum"),
        old_only=("old_only", "sum"),
        new_only=("new_only", "sum"),
    )
    .to_string()
)
