import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

DATA = Path("data/raw/xkdr/2024")
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

STATIONS = MANIFEST["stations"]
EXPECTED_COLUMNS = {
    "station_id", "parameter_name", "unit", "collected_at", "value"
}

station_rows = Counter()
station_months = Counter()
station_timestamps = defaultdict(list)
units = defaultdict(Counter)
invalid_values = Counter()
invalid_timestamps = Counter()
duplicate_keys = Counter()
unexpected_stations = Counter()
unexpected_parameters = Counter()
total_rows = 0
all_timestamp_formats_ok = True

for month in range(1, 13):
    path = DATA / f"2024-{month:02d}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(f"{path}: missing CSV header")

        missing = EXPECTED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"{path}: missing columns {missing}")

        for row in reader:
            total_rows += 1
            station = row["station_id"]
            timestamp_text = row["collected_at"]
            parameter = row["parameter_name"]
            unit = row["unit"]

            station_rows[station] += 1
            station_months[(station, month)] += 1
            units[station][unit] += 1

            if station not in STATIONS:
                unexpected_stations[station] += 1

            if parameter != "PM2.5":
                unexpected_parameters[parameter] += 1

            try:
                timestamp = datetime.fromisoformat(timestamp_text)
                station_timestamps[station].append(timestamp)
            except (ValueError, TypeError):
                invalid_timestamps[station] += 1
                all_timestamp_formats_ok = False
                continue

            try:
                value = float(row["value"])
                if not (-float("inf") < value < float("inf")):
                    invalid_values[station] += 1
            except (ValueError, TypeError):
                invalid_values[station] += 1

print("\n========== OVERALL ==========")
print(f"CSV rows read: {total_rows:,}")
print(f"Manifest rows: {MANIFEST['total_rows']:,}")
print(f"Row count matches: {total_rows == MANIFEST['total_rows']}")
print(f"Unexpected stations: {dict(unexpected_stations)}")
print(f"Unexpected parameters: {dict(unexpected_parameters)}")

print("\n========== STATION COVERAGE ==========")
for station in STATIONS:
    timestamps = station_timestamps[station]
    unique_timestamps = set(timestamps)

    if timestamps:
        first = min(timestamps)
        last = max(timestamps)
    else:
        first = last = "NO VALID TIMESTAMPS"

    print(f"\n{station}")
    print(f"  Rows: {station_rows[station]:,}")
    print(f"  Unique timestamps: {len(unique_timestamps):,}")
    print(f"  Duplicate station/timestamp rows: "
          f"{len(timestamps) - len(unique_timestamps):,}")
    print(f"  First observation: {first}")
    print(f"  Last observation:  {last}")
    print(f"  Units: {dict(units[station])}")
    print(f"  Invalid timestamps: {invalid_timestamps[station]}")
    print(f"  Invalid values: {invalid_values[station]}")

print("\n========== MONTHLY STATION COUNTS ==========")
print("Station       " + " ".join(f"{m:>5}" for m in range(1, 13)))

for station in STATIONS:
    counts = [
        station_months[(station, month)]
        for month in range(1, 13)
    ]
    print(f"{station:<13}" + " ".join(f"{n:>5}" for n in counts))

print("\n========== TIMESTAMP INTERVALS ==========")
for station in STATIONS:
    timestamps = sorted(set(station_timestamps[station]))
    intervals = Counter()

    for a, b in zip(timestamps, timestamps[1:]):
        minutes = (b - a).total_seconds() / 60
        if minutes > 0:
            intervals[round(minutes)] += 1

    print(f"\n{station}")
    print(f"  Most common intervals (minutes): {intervals.most_common(5)}")

print("\n========== AUDIT FLAGS ==========")
print(f"Invalid timestamp formats: {sum(invalid_timestamps.values())}")
print(f"Invalid numeric values: {sum(invalid_values.values())}")
print(f"Unexpected station rows: {sum(unexpected_stations.values())}")
print(f"Unexpected pollutant rows: {sum(unexpected_parameters.values())}")

print("\nAudit complete. No source CSVs were modified.")
