import pandas as pd
from pathlib import Path

op = pd.read_csv("data/raw/opencity/sion_mpcb_2024_25.csv")
xk = pd.concat(
    [pd.read_csv(p) for p in sorted(Path("data/raw/xkdr/2024").glob("*.csv"))],
    ignore_index=True
)

# Interpretation A: preserve the original clock reading as a naive timestamp.
op["wall_clock"] = pd.to_datetime(
    op["Timestamp"].astype(str).str.replace(r"\+0000$", "", regex=True)
)
op["PM2.5 (µg/m³)"] = pd.to_numeric(op["PM2.5 (µg/m³)"], errors="coerce")
op["PM10 (µg/m³)"] = pd.to_numeric(op["PM10 (µg/m³)"], errors="coerce")

xk["collected_at"] = pd.to_datetime(xk["collected_at"])
xk["value"] = pd.to_numeric(xk["value"], errors="coerce")
xk = xk[xk["station_id"] == "site_5119"].set_index("collected_at")

start = pd.Timestamp("2024-07-06 20:00:00")
end = pd.Timestamp("2024-07-07 18:00:00")

w = op[op["wall_clock"].between(start, end)].copy()
w = w[
    w["PM2.5 (µg/m³)"].notna() |
    (w["PM10 (µg/m³)"] >= 200)
].sort_values("wall_clock")

print("=== ORIGINAL CLOCK vs XKDR IST CLOCK ===")
for _, r in w.iterrows():
    t = r["wall_clock"]
    x = xk["value"].get(t, float("nan"))
    print(
        f'{t} | OpenCity PM2.5={r["PM2.5 (µg/m³)"]} '
        f'PM10={r["PM10 (µg/m³)"]} | XKDR={x}'
    )

print("\n=== EXACT-HOUR COMPARISON ===")
for t in [
    "2024-07-06 22:00",
    "2024-07-06 23:00",
    "2024-07-07 14:00",
    "2024-07-07 15:00"
]:
    ts = pd.Timestamp(t)
    rows = w[w["wall_clock"] == ts]
    print("\n", t)
    print(rows[["Timestamp", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]].to_string(index=False))
    print("XKDR:", xk["value"].get(ts, float("nan")))
