import pandas as pd
from pathlib import Path

op = pd.read_csv("data/raw/opencity/sion_mpcb_2024_25.csv")
xk = pd.concat(
    [pd.read_csv(p) for p in sorted(Path("data/raw/xkdr/2024").glob("*.csv"))],
    ignore_index=True
)

op = op[op["Station ID"] == "site_5119"].copy()
op["pm25"] = pd.to_numeric(op["PM2.5 (µg/m³)"], errors="coerce")

xk = xk[xk["station_id"] == "site_5119"].copy()
xk["value"] = pd.to_numeric(xk["value"], errors="coerce")
xk["time"] = pd.to_datetime(xk["collected_at"], errors="coerce")
xk = xk[["time", "value"]].dropna().rename(columns={"value": "xkdr"})

if xk["time"].duplicated().any():
    raise ValueError("Duplicate XKDR Sion timestamps found; resolve before comparison.")

op["time_utc_interpreted"] = (
    pd.to_datetime(op["Timestamp"], utc=True, errors="coerce")
      .dt.tz_convert("Asia/Kolkata")
      .dt.tz_localize(None)
)

op["time_wallclock_interpreted"] = pd.to_datetime(
    op["Timestamp"].astype(str).str.replace(r"\+0000$", "", regex=True),
    errors="coerce"
)

def report_metrics(label, merged, value_col):
    merged = merged.dropna(subset=[value_col, "xkdr"]).copy()
    merged["abs_error"] = (merged[value_col] - merged["xkdr"]).abs()

    print(f"\n{label}")
    print("Matched hours:", len(merged))

    if merged.empty:
        return

    print("Exact matches (<0.01):", (merged["abs_error"] < 0.01).sum())
    print("Median absolute difference:", merged["abs_error"].median())
    print("Correlation:", merged[[value_col, "xkdr"]].corr().iloc[0, 1])

    event = merged[
        merged["time"].between("2024-07-06 20:00", "2024-07-07 18:00")
    ]
    print("\nEvent comparison:")
    print(event.to_string(index=False))

for label, col in [
    ("A: UTC converted to IST", "time_utc_interpreted"),
    ("B: original clock treated as IST", "time_wallclock_interpreted")
]:
    print(f"\n{'=' * 12} {label} {'=' * 12}")

    candidate = op[["pm25", col]].dropna().rename(columns={col: "time"})

    duplicate_count = int(candidate["time"].duplicated(keep=False).sum())
    print("OpenCity rows with duplicated timestamps:", duplicate_count)

    # Original comparison: one reading exactly at the start of each hour.
    top_hour = candidate[candidate["time"].dt.minute == 0].copy()
    top_hour = top_hour.rename(columns={"pm25": "opencity_pm25"})
    top_hour = top_hour.merge(xk, on="time", how="inner")
    report_metrics("Top-of-hour sample comparison (not hourly mean):",
                   top_hour, "opencity_pm25")

    # Actual hourly mean: aggregate all 15-minute readings in each hour.
    hourly = (
        candidate.assign(hour=candidate["time"].dt.floor("h"))
        .groupby("hour")["pm25"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"hour": "time", "mean": "opencity_hourly_mean",
                         "count": "quarter_hour_readings"})
    )

    print("\n15-minute readings per OpenCity hour:")
    print(hourly["quarter_hour_readings"].value_counts().sort_index().to_string())

    # Only compare complete hours containing all four quarter-hour readings.
    complete = hourly[hourly["quarter_hour_readings"] == 4].copy()
    merged = complete.merge(xk, on="time", how="inner")

    report_metrics("Complete-hour mean comparison (4 readings per hour):",
                   merged, "opencity_hourly_mean")

print("\nTimestamp and hourly aggregation comparison complete.")
print("Neither source file was modified.")
