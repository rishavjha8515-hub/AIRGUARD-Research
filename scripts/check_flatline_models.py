
import pandas as pd

path = "reports/airguard_ml/station_model_comparison.csv"

df = pd.read_csv(path, parse_dates=["collected_at"])

cases = [
    {
        "name": "Sion",
        "station": "site_5119",
        "start": "2024-09-20 00:00",
        "end": "2024-09-22 21:00",
    },
    {
        "name": "Kurla",
        "station": "site_5104",
        "start": "2024-10-21 11:00",
        "end": "2024-10-24 13:00",
    },
]

print("=== ML DETECTION OF KNOWN FLATLINES ===")

for case in cases:
    start = pd.Timestamp(case["start"])
    end = pd.Timestamp(case["end"])

    mask = (
        (df["station_id"] == case["station"])
        & (df["collected_at"] >= start)
        & (df["collected_at"] <= end)
    )

    subset = df.loc[mask]

    print(f"\n{case['name']}")
    print("Observations:", len(subset))

    if subset.empty:
        print("No matching observations found.")
        continue

    print("Global flags:", int(subset["global_ml_flag"].sum()))
    print("Station-specific flags:", int(subset["station_ml_flag"].sum()))

    agreement = (
        subset["global_ml_flag"]
        & subset["station_ml_flag"]
    )

    print("Both models flag:", int(agreement.sum()))

    print(
        "Global flag percentage:",
        round(subset["global_ml_flag"].mean() * 100, 2),
        "%"
    )

    print(
        "Station flag percentage:",
        round(subset["station_ml_flag"].mean() * 100, 2),
        "%"
    )
