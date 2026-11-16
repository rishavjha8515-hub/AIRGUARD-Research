from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/xkdr/2024")
SCORES = Path("reports/temporal_baseline/temporal_scores.csv")
OUT = Path("reports/temporal_baseline")

raw = pd.concat(
    [pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))],
    ignore_index=True
)

scores = pd.read_csv(SCORES)

raw["collected_at"] = pd.to_datetime(raw["collected_at"])
scores["collected_at"] = pd.to_datetime(scores["collected_at"])

raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
scores["baseline_mad"] = pd.to_numeric(
    scores["baseline_mad"], errors="coerce"
)
scores["baseline_median"] = pd.to_numeric(
    scores["baseline_median"], errors="coerce"
)

targets = scores[
    scores["baseline_mad"].eq(0)
    & scores["baseline_median"].notna()
].copy()

results = []

for station, station_targets in targets.groupby("station_id"):
    history = raw[raw["station_id"] == station].sort_values(
        "collected_at"
    )

    for _, target in station_targets.iterrows():
        t = target["collected_at"]

        window = history[
            (history["collected_at"] >= t - pd.Timedelta(hours=24))
            & (history["collected_at"] < t)
        ]["value"].dropna()

        median = target["baseline_median"]
        deviations = (window - median).abs()

        results.append({
            "station_id": station,
            "collected_at": t,
            "current_value": target["value"],
            "baseline_median": median,
            "stored_baseline_count": target["baseline_count"],
            "reconstructed_count": len(window),
            "unique_values": window.nunique(),
            "median_frequency": int(
                np.isclose(window, median, atol=1e-9).sum()
            ),
            "median_fraction": (
                float(np.isclose(window, median, atol=1e-9).mean())
                if len(window) else np.nan
            ),
            "window_min": window.min(),
            "window_max": window.max(),
            "window_iqr": (
                window.quantile(0.75) - window.quantile(0.25)
                if len(window) else np.nan
            ),
            "window_std": window.std(),
            "reconstructed_mad": deviations.median(),
        })

result = pd.DataFrame(results)
result.to_csv(OUT / "zero_mad_window_audit.csv", index=False)

print("=== ZERO-MAD WINDOW AUDIT ===")
print("Windows inspected:", len(result))
print("Count mismatches:",
      int((result["stored_baseline_count"]
           != result["reconstructed_count"]).sum()))

print("\n=== STATION SUMMARY ===")
print(
    result.groupby("station_id").agg(
        windows=("collected_at", "size"),
        median_unique_values=("unique_values", "median"),
        median_median_fraction=("median_fraction", "median"),
        median_iqr=("window_iqr", "median"),
        max_iqr=("window_iqr", "max"),
    ).to_string()
)

print("\n=== ZERO-MAD WINDOWS WITH MOST VARIATION ===")
print(
    result.sort_values("window_iqr", ascending=False)[[
        "station_id", "collected_at", "current_value",
        "baseline_median", "reconstructed_count",
        "unique_values", "median_frequency",
        "median_fraction", "window_iqr",
        "window_min", "window_max"
    ]].head(25).to_string(index=False)
)

print("\nSaved:", (OUT / "zero_mad_window_audit.csv").resolve())