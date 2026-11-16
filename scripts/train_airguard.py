
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

RAW_DIR = Path("data/raw/xkdr/2024")
OUT_DIR = Path("reports/airguard_ml")

TRAIN_END = "2024-08-31 23:00:00"
CONTAMINATION = 0.01
RANDOM_STATE = 42

FEATURES = [
    "value",
    "past_median",
    "past_mad",
    "robust_deviation",
    "absolute_change",
    "change_rate",
    "repeated_run",
]


def load_data():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No data files in {RAW_DIR}")

    frames = []
    for path in files:
        df = pd.read_csv(
            path,
            usecols=["station_id", "parameter_name",
                     "collected_at", "value", "unit"],
        )
        df["source_file"] = path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data["collected_at"] = pd.to_datetime(
        data["collected_at"], errors="coerce"
    )
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    if data[["station_id", "collected_at", "value", "unit"]].isna().any().any():
        raise ValueError("Invalid required observations.")

    if data["parameter_name"].nunique() != 1 or not (
        data["parameter_name"] == "PM2.5"
    ).all():
        raise ValueError("Expected PM2.5 observations only.")

    if data["unit"].nunique() != 1:
        raise ValueError("Mixed units detected.")

    if data.duplicated(["station_id", "collected_at"]).any():
        raise ValueError("Duplicate station/timestamp observations.")

    return data.sort_values(["station_id", "collected_at"])


def build_features(data):
    outputs = []

    for station, group in data.groupby("station_id", sort=True):
        group = group.sort_values("collected_at").copy()
        group = group.set_index("collected_at")

        past = group["value"].rolling(
            "24h", closed="left", min_periods=12
        )

        group["past_median"] = past.median()
        group["past_mad"] = past.apply(
            lambda x: np.median(np.abs(x - np.median(x))),
            raw=True,
        )

        scale = (1.4826 * group["past_mad"]).clip(lower=0.5)

        group["robust_deviation"] = (
            (group["value"] - group["past_median"]).abs() / scale
        )

        group["absolute_change"] = group["value"].diff().abs()
        group["change_rate"] = group["value"].diff()

        # Consecutive exact repeated values, computed using past data.
        previous = group["value"].shift(1)
        same_as_previous = group["value"].eq(previous)

        run = []
        count = 0
        for repeated in same_as_previous.fillna(False):
            count = count + 1 if repeated else 0
            run.append(count)

        group["repeated_run"] = run

        outputs.append(group.reset_index())

    return pd.concat(outputs, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    features = build_features(data)

    train_end = pd.Timestamp(TRAIN_END)

    train = features.loc[
        features["collected_at"] <= train_end
    ].copy()

    test = features.loc[
        features["collected_at"] > train_end
    ].copy()

    train = train.dropna(subset=FEATURES)
    test = test.dropna(subset=FEATURES)

    if len(train) < 1000 or len(test) < 100:
        raise ValueError("Insufficient training or test observations.")

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(train[FEATURES])

    test["anomaly_score"] = -model.decision_function(test[FEATURES])
    test["ml_candidate"] = model.predict(test[FEATURES]) == -1

    # Existing MAD baseline, recomputed from the same past-only features.
    test["mad_candidate"] = test["robust_deviation"] >= 6.0

    test["screening_status"] = np.where(
        test["ml_candidate"] & test["mad_candidate"],
        "both_methods_flag",
        np.where(
            test["ml_candidate"],
            "ml_only_review",
            np.where(
                test["mad_candidate"],
                "mad_only_review",
                "not_flagged",
            ),
        ),
    )

    output_path = OUT_DIR / "test_predictions.csv"
    test.to_csv(output_path, index=False)

    comparison = {
        "training_period_end": TRAIN_END,
        "test_period_start": str(test["collected_at"].min()),
        "test_period_end": str(test["collected_at"].max()),
        "training_observations": len(train),
        "test_observations": len(test),
        "stations": int(data["station_id"].nunique()),
        "features": FEATURES,
        "ml_candidates": int(test["ml_candidate"].sum()),
        "mad_candidates": int(test["mad_candidate"].sum()),
        "both_methods_flag": int(
            (test["ml_candidate"] & test["mad_candidate"]).sum()
        ),
        "ml_only": int(
            (test["ml_candidate"] & ~test["mad_candidate"]).sum()
        ),
        "mad_only": int(
            (~test["ml_candidate"] & test["mad_candidate"]).sum()
        ),
        "interpretation": (
            "Unsupervised anomaly screening, not confirmed sensor-fault "
            "classification. No observations were removed."
        ),
    }

    (OUT_DIR / "metadata.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print("AIRGUARD ML BASELINE COMPLETE")
    print(json.dumps(comparison, indent=2))
    print("Predictions:", output_path.resolve())


if __name__ == "__main__":
    main()

