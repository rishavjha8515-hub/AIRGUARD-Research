
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/raw/xkdr/2024")
OUT_DIR = Path("reports/airguard_ml")

TRAIN_END = pd.Timestamp("2024-08-31 23:00:00")
TEST_START = pd.Timestamp("2024-09-01 00:00:00")

CONTAMINATION = 0.01
N_ESTIMATORS = 200
RANDOM_STATE = 42
MIN_TRAIN_ROWS = 500

FEATURES = [
    "value",
    "past_median",
    "past_mad",
    "robust_deviation",
    "absolute_change",
    "change_rate",
    "repeated_run",
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

df = pd.concat(
    [pd.read_csv(path) for path in files],
    ignore_index=True
)

required = {
    "collected_at",
    "station_id",
    "parameter_name",
    "value",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(
    subset=["collected_at", "station_id", "value"]
)

df = df[df["parameter_name"] == "PM2.5"].copy()
df = df.sort_values(["station_id", "collected_at"])

if df.duplicated(["station_id", "collected_at"]).any():
    raise ValueError("Duplicate station/timestamp observations found")


# --------------------------------------------------
# FEATURE ENGINEERING
# Matches train_airguard.py exactly
# --------------------------------------------------

feature_frames = []

for station_id, group in df.groupby("station_id"):
    group = group.sort_values("collected_at").copy()
    group = group.set_index("collected_at")

    values = group["value"]

    # Past-only rolling baseline.
    rolling = values.rolling(
        "24h",
        closed="left",
        min_periods=12
    )

    group["past_median"] = rolling.median()

    group["past_mad"] = rolling.apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True
    )

    # Same robust scale and deviation formula as original model.
    scale = (1.4826 * group["past_mad"]).clip(lower=0.5)

    group["robust_deviation"] = (
        (group["value"] - group["past_median"]).abs() / scale
    )

    # Same signed difference and absolute difference as original.
    group["absolute_change"] = group["value"].diff().abs()
    group["change_rate"] = group["value"].diff()

    # Same consecutive-repeat calculation as original.
    previous = group["value"].shift(1)
    same_as_previous = group["value"].eq(previous)

    run = []
    count = 0

    for repeated in same_as_previous.fillna(False):
        count = count + 1 if repeated else 0
        run.append(count)

    group["repeated_run"] = run

    group["station_id"] = station_id

    feature_frames.append(group.reset_index())

data = pd.concat(feature_frames, ignore_index=True)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=FEATURES)

train = data[data["collected_at"] <= TRAIN_END].copy()
test = data[data["collected_at"] >= TEST_START].copy()

if train.empty or test.empty:
    raise ValueError("Training or test period is empty")

print("=== DATA SPLIT ===")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("Training stations:", train["station_id"].nunique())
print("Test stations:", test["station_id"].nunique())


# --------------------------------------------------
# GLOBAL ISOLATION FOREST
# --------------------------------------------------

global_model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

global_model.fit(train[FEATURES])

test["global_anomaly_score"] = (
    -global_model.decision_function(test[FEATURES])
)

test["global_ml_flag"] = (
    global_model.predict(test[FEATURES]) == -1
)


# --------------------------------------------------
# STATION-SPECIFIC ISOLATION FORESTS
# --------------------------------------------------

test["station_anomaly_score"] = np.nan
test["station_ml_flag"] = False

station_results = []

for station_id in sorted(test["station_id"].unique()):
    station_train = train[
        train["station_id"] == station_id
    ]

    station_test_mask = (
        test["station_id"] == station_id
    )

    station_test = test.loc[station_test_mask]

    if len(station_train) < MIN_TRAIN_ROWS:
        print(
            f"Skipping {station_id}: "
            f"only {len(station_train)} training observations"
        )
        continue

    if station_test.empty:
        continue

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(station_train[FEATURES])

    scores = -model.decision_function(
        station_test[FEATURES]
    )

    flags = model.predict(
        station_test[FEATURES]
    ) == -1

    test.loc[
        station_test_mask,
        "station_anomaly_score"
    ] = scores

    test.loc[
        station_test_mask,
        "station_ml_flag"
    ] = flags

    station_results.append({
        "station_id": station_id,
        "training_observations": len(station_train),
        "test_observations": len(station_test),
        "global_flags": int(
            test.loc[station_test_mask, "global_ml_flag"].sum()
        ),
        "station_flags": int(flags.sum()),
    })


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

test["model_agreement"] = np.select(
    [
        test["global_ml_flag"] & test["station_ml_flag"],
        test["global_ml_flag"] & ~test["station_ml_flag"],
        ~test["global_ml_flag"] & test["station_ml_flag"],
    ],
    [
        "BOTH_FLAG",
        "GLOBAL_ONLY",
        "STATION_ONLY",
    ],
    default="NEITHER_FLAG"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUT_DIR / "station_model_comparison.csv"
test.to_csv(output_path, index=False)

summary = pd.DataFrame(station_results)

print("\n=== GLOBAL VS STATION-SPECIFIC MODEL ===")

print(
    "Global ML flags:",
    int(test["global_ml_flag"].sum())
)

print(
    "Station-specific ML flags:",
    int(test["station_ml_flag"].sum())
)

print("\n=== MODEL AGREEMENT ===")
print(test["model_agreement"].value_counts().to_string())

print("\n=== FLAGS BY STATION ===")

if not summary.empty:
    print(summary.to_string(index=False))

print("\n=== DEONAR COMPARISON ===")

deonar = summary[
    summary["station_id"] == "site_5396"
]

if not deonar.empty:
    print(deonar.to_string(index=False))
else:
    print("Deonar model was not fitted.")

print("\nSaved:", output_path)

print(
    "\nNote: These are unsupervised screening results, "
    "not confirmed sensor faults."
)
