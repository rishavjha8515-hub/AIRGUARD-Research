
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

WINDOW_LENGTH = 48
FAULT_TYPES = ["isolated_spike", "short_flatline", "stuck_at"]

# Conservatively exclude the injection and the following 24 hours.
CONTAMINATION_HOURS = 24


# --------------------------------------------------
# LOAD RAW DATA
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

df = df.sort_values(
    ["station_id", "collected_at"]
).reset_index(drop=True)

if df.duplicated(["station_id", "collected_at"]).any():
    raise ValueError("Duplicate station/timestamp observations found")

# Stable identifier for mapping modified observations to raw rows.
df["row_id"] = np.arange(len(df))


# --------------------------------------------------
# FEATURE ENGINEERING
# Exact formulas from train_airguard.py
# --------------------------------------------------

def engineer_features(source):
    frames = []

    for station_id, group in source.groupby("station_id"):
        group = group.sort_values("collected_at").copy()
        group = group.set_index("collected_at")

        values = group["value"]

        rolling = values.rolling(
            "24h",
            closed="left",
            min_periods=12
        )

        group["past_median"] = rolling.median()

        group["past_mad"] = rolling.apply(
            lambda x: np.median(
                np.abs(x - np.median(x))
            ),
            raw=True
        )

        scale = (
            1.4826 * group["past_mad"]
        ).clip(lower=0.5)

        group["robust_deviation"] = (
            (group["value"] - group["past_median"]).abs()
            / scale
        )

        group["absolute_change"] = group["value"].diff().abs()
        group["change_rate"] = group["value"].diff()

        previous = group["value"].shift(1)
        same_as_previous = group["value"].eq(previous)

        run = []
        count = 0

        for repeated in same_as_previous.fillna(False):
            count = count + 1 if repeated else 0
            run.append(count)

        group["repeated_run"] = run
        group["station_id"] = station_id

        frames.append(group.reset_index())

    result = pd.concat(frames, ignore_index=True)

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result


# --------------------------------------------------
# FIND BASELINE-CLEAN CONTIGUOUS HOURLY WINDOWS
# --------------------------------------------------

def find_windows(station_data, clean_row_ids):
    station_data = station_data.sort_values(
        "collected_at"
    ).reset_index(drop=True)

    timestamps = station_data["collected_at"]
    row_ids = station_data["row_id"]

    windows = []

    for start in range(
        len(station_data) - WINDOW_LENGTH + 1
    ):
        end = start + WINDOW_LENGTH

        segment_times = timestamps.iloc[start:end]
        segment_ids = row_ids.iloc[start:end].tolist()

        # Require 48 consecutive hourly observations.
        differences = segment_times.diff().dropna()

        if not (
            differences == pd.Timedelta(hours=1)
        ).all():
            continue

        # Every observation must have complete baseline features
        # and must be unflagged by all four baseline detectors.
        if not all(row_id in clean_row_ids for row_id in segment_ids):
            continue

        windows.append(segment_ids)

    return windows


def select_three_windows(windows):
    if len(windows) < 3:
        return None

    targets = [
        0.15 * (len(windows) - 1),
        0.50 * (len(windows) - 1),
        0.85 * (len(windows) - 1),
    ]

    selected = []

    for target in targets:
        available = [
            i for i in range(len(windows))
            if all(
                set(windows[i]).isdisjoint(set(previous))
                for previous in selected
            )
        ]

        if not available:
            return None

        chosen = min(
            available,
            key=lambda i: abs(i - target)
        )

        selected.append(windows[chosen])

    return selected


# --------------------------------------------------
# BASELINE FEATURES AND TRAINING DATA
# --------------------------------------------------

baseline_features = engineer_features(df)

baseline_features = baseline_features.dropna(
    subset=FEATURES
)

train = baseline_features[
    baseline_features["collected_at"] <= TRAIN_END
].copy()

test_baseline = baseline_features[
    baseline_features["collected_at"] >= TEST_START
].copy()

if train.empty or test_baseline.empty:
    raise ValueError("Training or test period is empty")

print("=== BASELINE DATA ===")
print("Training observations:", len(train))
print("Test observations:", len(test_baseline))
print("Training stations:", train["station_id"].nunique())
print("Test stations:", test_baseline["station_id"].nunique())


# --------------------------------------------------
# FIT MODELS ON ORIGINAL TRAINING DATA
# --------------------------------------------------

global_model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

global_model.fit(train[FEATURES])

station_models = {}

for station_id in sorted(train["station_id"].unique()):
    station_train = train[
        train["station_id"] == station_id
    ]

    if len(station_train) < MIN_TRAIN_ROWS:
        continue

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(station_train[FEATURES])
    station_models[station_id] = model


# --------------------------------------------------
# BASELINE DETECTOR FLAGS
# Identify windows that are clean before injection.
# --------------------------------------------------

baseline_eval = test_baseline.copy()

baseline_eval["global_flag"] = (
    global_model.predict(baseline_eval[FEATURES]) == -1
)

baseline_eval["station_flag"] = False

for station_id, model in station_models.items():
    mask = baseline_eval["station_id"] == station_id

    if mask.any():
        baseline_eval.loc[mask, "station_flag"] = (
            model.predict(
                baseline_eval.loc[mask, FEATURES]
            ) == -1
        )

baseline_eval["mad_flag"] = (
    baseline_eval["robust_deviation"] >= 6.0
)

baseline_eval["flatline_flag"] = (
    baseline_eval["repeated_run"] >= 11
)

baseline_detectors = [
    "global_flag",
    "station_flag",
    "mad_flag",
    "flatline_flag",
]

baseline_eval["baseline_any_flag"] = (
    baseline_eval[baseline_detectors].any(axis=1)
)

clean_row_ids = set(
    baseline_eval.loc[
        ~baseline_eval["baseline_any_flag"],
        "row_id"
    ]
)

print("\n=== BASELINE CLEAN WINDOW SCREEN ===")
print("Baseline-clean observations:", len(clean_row_ids))
print(
    "Baseline observations flagged by at least one detector:",
    int(baseline_eval["baseline_any_flag"].sum())
)


# --------------------------------------------------
# SELECT INJECTION WINDOWS
# --------------------------------------------------

test_raw = df[
    df["collected_at"] >= TEST_START
].copy()

station_windows = {}

for station_id, station_data in test_raw.groupby("station_id"):
    windows = find_windows(station_data, clean_row_ids)
    selected = select_three_windows(windows)

    if selected is None:
        print(
            f"Skipping {station_id}: "
            f"fewer than three non-overlapping "
            "baseline-clean 48-hour windows"
        )
        continue

    station_windows[station_id] = {
        "isolated_spike": selected[0],
        "short_flatline": selected[1],
        "stuck_at": selected[2],
    }

if not station_windows:
    raise ValueError(
        "No stations have three suitable baseline-clean windows"
    )

print("\n=== SELECTED INJECTION WINDOWS ===")
print("Stations selected:", len(station_windows))

for station_id, fault_windows in station_windows.items():
    for fault_type, row_ids in fault_windows.items():
        selected_rows = df[
            df["row_id"].isin(row_ids)
        ].sort_values("collected_at")

        print(
            station_id,
            fault_type,
            selected_rows["collected_at"].iloc[0],
            "to",
            selected_rows["collected_at"].iloc[-1]
        )


# --------------------------------------------------
# EVALUATE EACH FAULT TYPE
# --------------------------------------------------

results = []

for fault_type in FAULT_TYPES:

    print(f"\n=== TESTING {fault_type.upper()} ===")

    modified = df.copy()
    modified["synthetic_event_id"] = ""

    injected_rows = 0
    event_count = 0

    # Raw row IDs affected by injection and feature contamination.
    affected_row_ids = set()

    for station_id, fault_windows in station_windows.items():

        row_ids = fault_windows[fault_type]
        event_id = f"{station_id}_{fault_type}"

        positions = modified.index[
            modified["row_id"].isin(row_ids)
        ].tolist()

        if len(positions) != WINDOW_LENGTH:
            raise ValueError(
                f"Unexpected window size for {event_id}"
            )

        event_count += 1

        initial_value = modified.loc[
            positions[0], "value"
        ]

        if fault_type == "isolated_spike":

            start_features = baseline_features[
                baseline_features["row_id"] == row_ids[0]
            ]

            if start_features.empty:
                raise ValueError(
                    f"Missing baseline features for {event_id}"
                )

            baseline_median = start_features.iloc[0]["past_median"]
            baseline_mad = start_features.iloc[0]["past_mad"]

            scale = max(1.4826 * baseline_mad, 0.5)

            spike_value = baseline_median + 10 * scale

            # Inject one isolated high reading.
            injected_positions = positions[:1]

            modified.loc[
                injected_positions, "value"
            ] = spike_value

        elif fault_type == "short_flatline":

            # First 12 readings become identical.
            injected_positions = positions[:12]

            modified.loc[
                injected_positions, "value"
            ] = initial_value

        elif fault_type == "stuck_at":

            # All 48 readings become identical.
            injected_positions = positions

            modified.loc[
                injected_positions, "value"
            ] = initial_value

        modified.loc[
            injected_positions, "synthetic_event_id"
        ] = event_id

        injected_rows += len(injected_positions)

        # Determine the actual time span of modified observations.
        injected_times = modified.loc[
            injected_positions, "collected_at"
        ]

        first_injected_time = injected_times.min()
        last_injected_time = injected_times.max()

        # Exclude injected observations and subsequent observations
        # from the same station for the next 24 hours.
        affected = modified[
            (modified["station_id"] == station_id)
            & (
                modified["collected_at"] >= first_injected_time
            )
            & (
                modified["collected_at"]
                <= last_injected_time
                + pd.Timedelta(hours=CONTAMINATION_HOURS)
            )
        ]

        affected_row_ids.update(
            affected["row_id"].tolist()
        )

    # --------------------------------------------------
    # RECOMPUTE FEATURES AFTER INJECTION
    # --------------------------------------------------

    engineered = engineer_features(modified)

    engineered = engineered.dropna(
        subset=FEATURES
    )

    test = engineered[
        engineered["collected_at"] >= TEST_START
    ].copy()

    test["injected"] = (
        test["synthetic_event_id"] != ""
    )

    test["affected_by_injection"] = (
        test["row_id"].isin(affected_row_ids)
    )

    # Verify every injected observation remains in the evaluation.
    if int(test["injected"].sum()) != injected_rows:
        raise ValueError(
            f"Injected observation mismatch for {fault_type}: "
            f"expected {injected_rows}, "
            f"found {int(test['injected'].sum())}"
        )

    # --------------------------------------------------
    # RUN DETECTORS
    # --------------------------------------------------

    test["global_flag"] = (
        global_model.predict(test[FEATURES]) == -1
    )

    test["station_flag"] = False

    for station_id, model in station_models.items():
        mask = test["station_id"] == station_id

        if mask.any():
            test.loc[mask, "station_flag"] = (
                model.predict(test.loc[mask, FEATURES]) == -1
            )

    test["mad_flag"] = (
        test["robust_deviation"] >= 6.0
    )

    test["flatline_flag"] = (
        test["repeated_run"] >= 11
    )

    detectors = [
        "global_flag",
        "station_flag",
        "mad_flag",
        "flatline_flag",
    ]

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    injected = test[test["injected"]]

    # Exclude both injected rows and rows whose temporal features
    # could have been affected by an injection.
    untouched = test[
        ~test["affected_by_injection"]
    ]

    print("Injected observations:", len(injected))
    print("Excluded due to injection influence:",
          int(test["affected_by_injection"].sum()))
    print("Untouched observations:", len(untouched))

    for detector in detectors:

        row_recall = (
            injected[detector].mean()
            if len(injected) else np.nan
        )

        untouched_flag_count = int(
            untouched[detector].sum()
        )

        untouched_flag_rate = (
            untouched_flag_count / len(untouched)
            if len(untouched) else np.nan
        )

        detected_events = 0

        for event_id in injected["synthetic_event_id"].unique():
            event_rows = injected[
                injected["synthetic_event_id"] == event_id
            ]

            if event_rows[detector].any():
                detected_events += 1

        event_recall = (
            detected_events / event_count
            if event_count else np.nan
        )

        results.append({
            "fault_type": fault_type,
            "detector": detector,
            "events_injected": event_count,
            "injected_observations": len(injected),
            "detected_events": detected_events,
            "event_recall": event_recall,
            "row_recall": row_recall,
            "untouched_flag_rate": untouched_flag_rate,
            "untouched_flag_count": untouched_flag_count,
            "untouched_observations": len(untouched),
        })

        print(
            f"{detector}: "
            f"event recall={event_recall:.3f}, "
            f"row recall={row_recall:.3f}, "
            f"untouched flag rate={untouched_flag_rate:.4f}"
        )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)

output_path = OUT_DIR / "synthetic_fault_validation.csv"

results_df.to_csv(output_path, index=False)

print("\n=== SYNTHETIC FAULT VALIDATION SUMMARY ===")

print(
    results_df[
        [
            "fault_type",
            "detector",
            "events_injected",
            "injected_observations",
            "detected_events",
            "event_recall",
            "row_recall",
            "untouched_flag_rate",
            "untouched_observations",
        ]
    ].to_string(index=False)
)

print("\nSaved:", output_path)

print(
    "\nINTERPRETATION:"
    "\n- Event recall: fraction of injected events detected at least once."
    "\n- Row recall: fraction of injected observations flagged."
    "\n- Untouched flag rate: fraction of observations outside "
    "injection influence windows that were flagged."
    "\n- Untouched flag rate is NOT a true false-positive rate, "
    "because real-world fault labels are unavailable."
    "\n- Synthetic detection performance does not establish "
    "real-world sensor fault detection."
)
