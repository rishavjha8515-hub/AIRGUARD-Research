
from pathlib import Path
import pandas as pd

path = Path("reports/airguard_ml/test_predictions.csv")
df = pd.read_csv(path, parse_dates=["collected_at"])

print("=== OVERALL ===")
print("Rows:", len(df))
print("ML candidates:", df["ml_candidate"].sum())
print("MAD candidates:", df["mad_candidate"].sum())

print("\n=== BY STATION ===")
print(
    df.groupby("station_id").agg(
        observations=("value", "size"),
        ml_flags=("ml_candidate", "sum"),
        mad_flags=("mad_candidate", "sum"),
    ).to_string()
)

print("\n=== BY MONTH ===")
df["month"] = df["collected_at"].dt.month
print(
    df.groupby("month").agg(
        observations=("value", "size"),
        ml_flags=("ml_candidate", "sum"),
        mad_flags=("mad_candidate", "sum"),
    ).to_string()
)

print("\n=== TOP 20 ML ANOMALY SCORES ===")
print(
    df.nlargest(20, "anomaly_score")[
        ["station_id", "collected_at", "value",
         "past_median", "past_mad", "robust_deviation",
         "repeated_run", "anomaly_score",
         "ml_candidate", "mad_candidate"]
    ].to_string(index=False)
)

print("\n=== KNOWN FLATLINE: SION ===")
sion = df[
    (df["station_id"] == "site_5119")
    & df["collected_at"].between(
        "2024-09-20", "2024-09-23"
    )
]
print(sion[
    ["collected_at", "value", "repeated_run",
     "anomaly_score", "ml_candidate", "mad_candidate"]
].to_string(index=False))

print("\n=== KNOWN FLATLINE: KURLA ===")
kurla = df[
    (df["station_id"] == "site_5104")
    & df["collected_at"].between(
        "2024-10-21", "2024-10-25"
    )
]
print(kurla[
    ["collected_at", "value", "repeated_run",
     "anomaly_score", "ml_candidate", "mad_candidate"]
].to_string(index=False))
