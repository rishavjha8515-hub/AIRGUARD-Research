
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/xkdr/2024")
OUT_DIR = Path("reports/temporal_baseline")

WINDOW = "24h"
MIN_OBSERVATIONS = 12
THRESHOLD = 6.0


def main():
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No monthly CSV files found in {RAW_DIR}")

    frames = []

    for path in files:
        frame = pd.read_csv(
            path,
            usecols=["station_id", "collected_at", "value", "unit"],
        )
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    data["collected_at"] = pd.to_datetime(
        data["collected_at"], errors="coerce"
    )
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    if data[["station_id", "collected_at", "value", "unit"]].isna().any().any():
        raise ValueError("Missing or invalid required observations. Stop and audit.")

    if data.duplicated(["station_id", "collected_at"]).any():
        raise ValueError("Duplicate station/timestamp observations found.")

    results = []

    for station, group in data.groupby("station_id", sort=True):
        group = group.sort_values("collected_at").set_index("collected_at")

        # closed='left' excludes the current observation.
        rolling = group["value"].rolling(
            WINDOW,
            closed="left",
            min_periods=MIN_OBSERVATIONS,
        )

        group["baseline_median"] = rolling.median()

        group["baseline_mad"] = rolling.apply(
            lambda x: np.median(np.abs(x - np.median(x))),
            raw=True,
        )

        group["baseline_count"] = rolling.count()

        robust_scale = 1.4826 * group["baseline_mad"]

        group["robust_z"] = (
            (group["value"] - group["baseline_median"]).abs()
            / robust_scale.where(robust_scale > 0)
        )

        group["candidate_anomaly"] = (
            group["robust_z"] >= THRESHOLD
        )

        group["screening_status"] = np.where(
            group["candidate_anomaly"],
            "candidate_anomaly_review",
            np.where(
                group["robust_z"].isna(),
                "insufficient_or_zero_MAD",
                "not_flagged_by_temporal_baseline",
            ),
        )

        results.append(group.reset_index())

    output = pd.concat(results, ignore_index=True)
    output = output.sort_values(["station_id", "collected_at"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUT_DIR / "temporal_scores.csv", index=False)

    candidates = output[output["candidate_anomaly"]].copy()
    candidates.to_csv(
        OUT_DIR / "temporal_review_candidates.csv",
        index=False,
    )

    metadata = {
        "method": "Past-only rolling median and MAD",
        "input_directory": str(RAW_DIR),
        "input_files": [p.name for p in files],
        "observations": len(output),
        "stations": int(output["station_id"].nunique()),
        "window": WINDOW,
        "minimum_observations": MIN_OBSERVATIONS,
        "threshold_robust_z": THRESHOLD,
        "candidate_count": len(candidates),
        "interpretation": (
            "Screening flags only. A temporal outlier is not proof "
            "of sensor failure. No observations were removed."
        ),
    }

    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("TEMPORAL BASELINE COMPLETE")
    print("Observations:", len(output))
    print("Stations:", output["station_id"].nunique())
    print("Candidate anomalies:", len(candidates))
    print("Unscored observations:", output["robust_z"].isna().sum())
    print("\nCandidates by station:")
    print(
        candidates.groupby("station_id").size().to_string()
        if not candidates.empty else "None"
    )
    print("\nOutput directory:", OUT_DIR.resolve())
    print("Raw observations were not modified.")


if __name__ == "__main__":
    main()

