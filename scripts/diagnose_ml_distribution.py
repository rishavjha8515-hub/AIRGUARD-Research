
import pandas as pd

path = "reports/airguard_ml/test_predictions.csv"
df = pd.read_csv(path)

print("=== TEST PERIOD STATION DISTRIBUTIONS ===")
print(
    df.groupby("station_id").agg(
        observations=("value", "size"),
        median=("value", "median"),
        p95=("value", lambda x: x.quantile(0.95)),
        max_value=("value", "max"),
        median_robust_deviation=("robust_deviation", "median"),
        ml_flags=("ml_candidate", "sum"),
    ).round(2).to_string()
)

print("\n=== DEONAR MONTHLY DISTRIBUTION ===")
deonar = df[df["station_id"] == "site_5396"].copy()
deonar["month"] = pd.to_datetime(
    deonar["collected_at"]
).dt.month

print(
    deonar.groupby("month").agg(
        observations=("value", "size"),
        median=("value", "median"),
        p95=("value", lambda x: x.quantile(0.95)),
        maximum=("value", "max"),
        flags=("ml_candidate", "sum"),
    ).round(2).to_string()
)
