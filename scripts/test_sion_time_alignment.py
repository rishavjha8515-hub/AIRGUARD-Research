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
xk["collected_at"] = pd.to_datetime(xk["collected_at"])

# Interpretation A: +0000 is a genuine UTC offset.
op["time_utc_interpreted"] = (
    pd.to_datetime(op["Timestamp"], utc=True)
      .dt.tz_convert("Asia/Kolkata")
      .dt.tz_localize(None)
)

# Interpretation B: the displayed clock is already IST.
op["time_wallclock_interpreted"] = pd.to_datetime(
    op["Timestamp"].astype(str).str.replace(
        r"\+0000$", "", regex=True
    )
)

xk = xk.rename(columns={"collected_at": "time", "value": "xkdr"})
xk = xk[["time", "xkdr"]].dropna()

for label, col in [
    ("A: UTC converted to IST", "time_utc_interpreted"),
    ("B: original clock treated as IST", "time_wallclock_interpreted")
]:
    candidate = op[["pm25", col]].rename(columns={col: "time"})
    candidate = candidate.dropna()
    candidate = candidate[candidate["time"].dt.minute == 0]
    candidate = candidate.drop_duplicates("time")

    merged = candidate.merge(xk, on="time", how="inner")
    merged["abs_error"] = (merged["pm25"] - merged["xkdr"]).abs()

    print(f"\n=== {label} ===")
    print("Matched hourly observations:", len(merged))
    print("Exact matches:", (merged["abs_error"] < 0.01).sum())

    if len(merged):
        print("Median absolute difference:", merged["abs_error"].median())
        print("Correlation:", merged[["pm25", "xkdr"]].corr().iloc[0, 1])

        event = merged[
            merged["time"].between(
                "2024-07-06 20:00", "2024-07-07 18:00"
            )
        ]
        print("\nEvent comparison:")
        print(event.to_string(index=False))

