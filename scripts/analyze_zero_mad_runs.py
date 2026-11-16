from pathlib import Path
import pandas as pd

path = Path("reports/temporal_baseline/zero_mad_window_audit.csv")
df = pd.read_csv(path)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df = df.sort_values(["station_id", "collected_at"])

# A window is classified as fully constant only when
# every reconstructed historical value equals the median.
df["fully_constant"] = (
    df["unique_values"] == 1
)

# Consecutive timestamps identify runs of zero-MAD windows.
df["time_gap"] = (
    df.groupby("station_id")["collected_at"].diff()
)

df["new_run"] = (
    df["time_gap"].isna()
    | (df["time_gap"] != pd.Timedelta(hours=1))
)

df["run_id"] = (
    df.groupby("station_id")["new_run"].cumsum()
)

runs = (
    df.groupby(["station_id", "run_id"])
    .agg(
        start=("collected_at", "min"),
        end=("collected_at", "max"),
        windows=("collected_at", "size"),
        fully_constant_windows=("fully_constant", "sum"),
        median_fraction=("median_fraction", "median"),
        median_iqr=("window_iqr", "median"),
        max_iqr=("window_iqr", "max"),
    )
    .reset_index()
)

runs["duration_hours"] = runs["windows"]

print("=== ZERO-MAD RUN SUMMARY ===")
print(
    runs.groupby("station_id").agg(
        runs=("run_id", "count"),
        longest_run_hours=("duration_hours", "max"),
        runs_3h_or_longer=(
            "duration_hours",
            lambda x: int((x >= 3).sum())
        ),
        fully_constant_windows=("fully_constant_windows", "sum"),
    ).to_string()
)

print("\n=== LONGEST ZERO-MAD RUNS ===")
print(
    runs.sort_values("duration_hours", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\n=== FULLY CONSTANT WINDOWS ===")
print(
    df.groupby("station_id").agg(
        zero_mad_windows=("collected_at", "size"),
        fully_constant=("fully_constant", "sum"),
        fully_constant_percent=(
            "fully_constant",
            lambda x: round(100 * x.mean(), 2)
        )
    ).to_string()
)

out = Path("reports/temporal_baseline")
runs.to_csv(out / "zero_mad_runs.csv", index=False)

print("\nSaved:", (out / "zero_mad_runs.csv").resolve())