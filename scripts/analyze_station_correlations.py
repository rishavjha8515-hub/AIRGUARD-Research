import csv
import math
from pathlib import Path
from itertools import combinations
from datetime import datetime, timedelta

ROOT = Path("data/raw/xkdr/2024")
data = {}

for path in sorted(ROOT.glob("2024-*.csv")):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["station_id"].strip()
            timestamp = datetime.fromisoformat(row["collected_at"])
            value = float(row["value"])
            data.setdefault(sid, {})[timestamp] = value

def correlation(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    dx = [x-mx for x in xs]
    dy = [y-my for y in ys]
    vx = sum(x*x for x in dx)
    vy = sum(y*y for y in dy)
    if vx == 0 or vy == 0:
        return None
    return sum(x*y for x, y in zip(dx, dy)) / math.sqrt(vx*vy)

results = []

for a, b in combinations(sorted(data), 2):
    shared = sorted(set(data[a]) & set(data[b]))

    levels_a = [data[a][t] for t in shared]
    levels_b = [data[b][t] for t in shared]
    level_corr = correlation(levels_a, levels_b)

    changes_a, changes_b = [], []
    for t in shared:
        previous = t - timedelta(hours=1)
        if previous in data[a] and previous in data[b]:
            changes_a.append(data[a][t] - data[a][previous])
            changes_b.append(data[b][t] - data[b][previous])

    change_corr = correlation(changes_a, changes_b)

    results.append((
        a, b, len(shared), level_corr,
        len(changes_a), change_corr
    ))

print("========== PAIRWISE TEMPORAL CORRELATION ==========")
print("Pearson correlations; hourly changes require consecutive shared hours.")
print(f"{'Pair':<25} {'N levels':>9} {'Level r':>10} {'N changes':>11} {'Change r':>10}")

for a, b, n, r1, nd, r2 in sorted(
    results, key=lambda x: (x[5] is None, -(x[5] or -2))
):
    print(
        f"{a.replace('site_', '')}-{b.replace('site_', ''):<17}"
        f"{n:>9} "
        f"{r1 if r1 is not None else float('nan'):>10.3f} "
        f"{nd:>11} "
        f"{r2 if r2 is not None else float('nan'):>10.3f}"
    )

print("\n========== INTERPRETATION REMINDER ==========")
print("High correlation is not proof of a shared pollution event.")
print("Low correlation is not proof of a faulty sensor.")
print("These are exploratory comparisons, not validated fault labels.")
