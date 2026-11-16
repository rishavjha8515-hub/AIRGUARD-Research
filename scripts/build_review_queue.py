
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ML_PATH = Path("reports/airguard_ml/test_predictions.csv")
MAD_PATH = Path("reports/temporal_baseline/temporal_scores.csv")
FLATLINE_PATH = Path("reports/airguard_ml/flatline_episodes.csv")

OUT_DIR = Path("reports/airguard_ml")
OUT_PATH = OUT_DIR / "unified_review_queue.csv"

KEYS = ["station_id", "collected_at"]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

for path in [ML_PATH, MAD_PATH, FLATLINE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

ml = pd.read_csv(ML_PATH, parse_dates=["collected_at"])
mad = pd.read_csv(MAD_PATH, parse_dates=["collected_at"])
episodes = pd.read_csv(
    FLATLINE_PATH,
    parse_dates=["start", "end"]
)

print("Loaded ML predictions:", len(ml))
print("Loaded MAD scores:", len(mad))
print("Loaded flatline episodes:", len(episodes))


# --------------------------------------------------
# VALIDATE INPUTS
# --------------------------------------------------

required_ml = {
    "collected_at", "station_id", "value",
    "ml_candidate", "mad_candidate"
}

required_mad = {
    "collected_at", "station_id",
    "baseline_median", "baseline_mad",
    "baseline_count", "robust_z",
    "candidate_anomaly"
}

required_flatline = {
    "station_id", "start", "end",
    "duration_hours", "value"
}

for name, df, required in [
    ("ML", ml, required_ml),
    ("MAD", mad, required_mad),
    ("Flatline", episodes, required_flatline),
]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} file missing columns: {sorted(missing)}")

if ml.duplicated(KEYS).any():
    raise ValueError("Duplicate station/timestamp keys in ML predictions")

if mad.duplicated(KEYS).any():
    raise ValueError("Duplicate station/timestamp keys in MAD scores")

if (episodes["end"] < episodes["start"]).any():
    raise ValueError("Flatline episode has end before start")


# --------------------------------------------------
# MERGE ML AND MAD RESULTS
# --------------------------------------------------

mad_fields = [
    "baseline_median",
    "baseline_mad",
    "baseline_count",
    "robust_z",
    "candidate_anomaly",
]

queue = ml.merge(
    mad[KEYS + mad_fields],
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator=True
)

unmatched_mad = (queue["_merge"] == "left_only").sum()
queue = queue.drop(columns="_merge")

print("ML observations without matching MAD records:", unmatched_mad)


# --------------------------------------------------
# EXPAND FLATLINE EPISODES
# --------------------------------------------------

flatline_rows = []

for _, episode in episodes.iterrows():
    timestamps = pd.date_range(
        start=episode["start"],
        end=episode["end"],
        freq="h"
    )

    for timestamp in timestamps:
        flatline_rows.append({
            "station_id": episode["station_id"],
            "collected_at": timestamp,
            "flatline_candidate": True,
            "flatline_value": episode["value"],
            "flatline_consecutive_readings": episode["duration_hours"],
        })

flatlines = pd.DataFrame(
    flatline_rows,
    columns=KEYS + [
        "flatline_candidate",
        "flatline_value",
        "flatline_consecutive_readings",
    ]
)

if not flatlines.empty:
    if flatlines.duplicated(KEYS).any():
        raise ValueError(
            "Overlapping flatline episodes found for the same observation"
        )

    queue = queue.merge(
        flatlines,
        on=KEYS,
        how="left",
        validate="one_to_one"
    )
else:
    queue["flatline_candidate"] = False
    queue["flatline_value"] = pd.NA
    queue["flatline_consecutive_readings"] = pd.NA


# --------------------------------------------------
# NORMALIZE SIGNAL COLUMNS
# --------------------------------------------------

for column in ["ml_candidate", "mad_candidate", "flatline_candidate"]:
    queue[column] = (
        queue[column]
        .fillna(False)
        .astype(bool)
    )

queue["review_required"] = (
    queue["ml_candidate"]
    | queue["mad_candidate"]
    | queue["flatline_candidate"]
)


# --------------------------------------------------
# EXPLAIN DETECTION SIGNALS
# --------------------------------------------------

def describe_signals(row):
    signals = []

    if row["ml_candidate"]:
        signals.append("isolation_forest")

    if row["mad_candidate"]:
        signals.append("robust_mad")

    if row["flatline_candidate"]:
        signals.append("flatline_pattern")

    return ";".join(signals) if signals else "none"


queue["review_signals"] = queue.apply(
    describe_signals,
    axis=1
)


# --------------------------------------------------
# ASSIGN REVIEW CATEGORY
# --------------------------------------------------

def classify_review(row):
    signal_count = sum([
        row["ml_candidate"],
        row["mad_candidate"],
        row["flatline_candidate"],
    ])

    if signal_count >= 2:
        return "MULTI_SIGNAL"

    if row["flatline_candidate"]:
        return "FLATLINE_PATTERN"

    if row["ml_candidate"] or row["mad_candidate"]:
        return "STATISTICAL_OUTLIER"

    return "NO_FLAG"


queue["review_category"] = queue.apply(
    classify_review,
    axis=1
)


# --------------------------------------------------
# BUILD REVIEW QUEUE
# --------------------------------------------------

review = queue.loc[queue["review_required"]].copy()

review = review.sort_values(
    ["station_id", "collected_at"]
).reset_index(drop=True)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n=== AIRGUARD UNIFIED REVIEW QUEUE ===")

print(f"Test observations: {len(queue)}")
print(f"Review candidates: {len(review)}")
print(f"Review percentage: {len(review) / len(queue) * 100:.2f}%")

print("\n=== INDIVIDUAL SIGNAL COUNTS ===")
print("Isolation Forest:", int(review["ml_candidate"].sum()))
print("Robust MAD:", int(review["mad_candidate"].sum()))
print("Flatline observations:", int(review["flatline_candidate"].sum()))

print("\n=== REVIEW CATEGORIES ===")
print(
    queue["review_category"]
    .value_counts()
    .to_string()
)

print("\n=== SIGNAL COMBINATIONS ===")
print(
    review["review_signals"]
    .value_counts()
    .to_string()
)

print("\n=== REVIEW CATEGORIES BY STATION ===")
print(
    pd.crosstab(
        queue["station_id"],
        queue["review_category"]
    ).to_string()
)

print("\n=== CANDIDATES BY STATION ===")
print(
    review.groupby("station_id")
    .agg(
        observations=("value", "size"),
        ml_flags=("ml_candidate", "sum"),
        mad_flags=("mad_candidate", "sum"),
        flatline_flags=("flatline_candidate", "sum"),
    )
    .to_string()
)


# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

review.to_csv(OUT_PATH, index=False)

print("\nSaved unified review queue:")
print(OUT_PATH)

print("\nNote: These are screening indicators, not confirmed")
print("sensor failures. No original measurements are deleted.")
