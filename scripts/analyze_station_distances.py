import json
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

path = Path("data/raw/xkdr_metadata/stations.response.txt")
payload = json.loads(path.read_text(encoding="utf-8"))

cohort = {
    "site_5113", "site_5399", "site_5120",
    "site_5396", "site_5104", "site_5112",
    "site_5119", "site_5106", "site_5115"
}

stations = {
    x["station_id"]: x
    for x in payload["data"]
    if x["station_id"] in cohort
}

def distance(a, b):
    lat1, lon1 = map(radians, (a["latitude"], a["longitude"]))
    lat2, lon2 = map(radians, (b["latitude"], b["longitude"]))
    dlat, dlon = lat2-lat1, lon2-lon1
    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(h))

print("========== STATION DISTANCES (KM) ==========")
ids = sorted(stations)

print(f"{'Station':<14}", end="")
for sid in ids:
    print(f"{sid.replace('site_', ''):>9}", end="")
print()

for a in ids:
    print(f"{a:<14}", end="")
    for b in ids:
        value = 0 if a == b else distance(stations[a], stations[b])
        print(f"{value:>9.2f}", end="")
    print()

print("\n========== NEAREST NEIGHBOURS ==========")
for sid in ids:
    s = stations[sid]
    neighbours = sorted(
        (distance(s, other), other_id)
        for other_id, other in stations.items()
        if other_id != sid
    )
    print(f"\n{sid} — {s['station_name']}")
    for km, other_id in neighbours[:4]:
        print(f"  {other_id}: {km:.2f} km — {stations[other_id]['station_name']}")

print("\n========== COORDINATE DUPLICATES ==========")
all_stations = payload["data"]
for i, a in enumerate(all_stations):
    if a.get("latitude") is None or a.get("longitude") is None:
        continue
    for b in all_stations[i+1:]:
        if (a.get("latitude"), a.get("longitude")) == (
            b.get("latitude"), b.get("longitude")
        ):
            print(f"{a['station_id']} / {b['station_id']}: "
                  f"{a['latitude']}, {a['longitude']}")

print("\nDistance analysis complete. No source data modified.")
