
from pathlib import Path
import pandas as pd

path = Path("data/raw/opencity/sion_mpcb_2024_25.csv")
df = pd.read_csv(path)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"].astype(str).str.replace(
        r"\+0000$", "", regex=True
    ),
    errors="coerce"
)

start = pd.Timestamp("2024-09-20 00:00:00")
end = pd.Timestamp("2024-09-22 21:15:00")

event = df.loc[
    (df["Station ID"] == "site_5119")
    & (df["Timestamp"] >= start)
    & (df["Timestamp"] <= end)
].copy()

measurement_columns = [
    c for c in event.columns
    if c not in [
        "Station ID", "State", "City",
        "Station Name", "Timestamp"
    ]
]

print("Rows:", len(event))
print("\n=== MEASUREMENT BEHAVIOUR DURING PM2.5 FLATLINE ===")

for col in measurement_columns:
    values = pd.to_numeric(event[col], errors="coerce")
    valid = values.dropna()

    if len(valid) == 0:
        continue

    print(
        f"{col:25} "
        f"valid={len(valid):3} "
        f"missing={values.isna().sum():3} "
        f"unique={valid.nunique():4} "
        f"min={valid.min():10.3f} "
        f"max={valid.max():10.3f}"
    )
