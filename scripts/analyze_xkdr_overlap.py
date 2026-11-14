import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from itertools import combinations

ROOT = Path("data/raw/xkdr/2024")
station_times = defaultdict(set)

for path in sorted(ROOT.glob("2024-*.csv")):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            station = row["station_id"].strip()
            timestamp = datetime.fromisoformat(row["collected_at"])
            station_times[station].add(timestamp)

stations = sorted(station_times)

print("========== PAIRWISE SIMULTANEOUS COVERAGE ==========")
print("Each value = number of timestamps shared by two stations")
print(f"{'Station':<14}", end="")
for s in stations:
    print(f"{s.replace('site_', ''):>9}", end="")
print()

for a in stations:
    print(f"{a:<14}", end="")
    for b in stations:
        n = len(station_times[a] & station_times[b])
        print(f"{n:>9}", end="")
    print()

print("\n========== HOURS WITH MULTIPLE STATIONS ==========")
hour_counts = Counter()
for timestamps in station_times.values():
    hour_counts.update(timestamps)

for n in range(1, len(stations) + 1):
    count = sum(1 for v in hour_counts.values() if v == n)
    print(f"Hours observed by exactly {n} stations: {count}")

print("\n========== MONTHLY MULTI-STATION COVERAGE ==========")
monthly = defaultdict(Counter)
for timestamp, count in hour_counts.items():
    monthly[timestamp.strftime("%Y-%m")][count] += 1

for month in sorted(monthly):
    c = monthly[month]
    print(
        f"{month}: "
        f">=2 stations={sum(v for k, v in c.items() if k >= 2)}, "
        f">=3 stations={sum(v for k, v in c.items() if k >= 3)}, "
        f">=5 stations={sum(v for k, v in c.items() if k >= 5)}, "
        f">=7 stations={sum(v for k, v in c.items() if k >= 7)}"
    )

print("\n========== LONGEST OBSERVATION GAPS ==========")
for station in stations:
    ts = sorted(station_times[station])
    gaps = []
    for previous, current in zip(ts, ts[1:]):
        delta = current - previous
        if delta > timedelta(hours=1):
            gaps.append((delta, previous, current))

    gaps.sort(reverse=True)
    print(f"\n{station}: {len(gaps)} gaps longer than one hour")
    for delta, previous, current in gaps[:5]:
        print(f"  {delta}: {previous} -> {current}")

print("\nCoverage analysis complete. Raw files were not modified.")

