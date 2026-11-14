import csv
import io
import json
import os
import time
from pathlib import Path
from datetime import date
from calendar import monthrange

import requests

BASE_URL = "https://airquality.xkdr.org/v1/measurements"

STATIONS = [
    "site_5113", "site_5399", "site_5120",
    "site_5396", "site_5104", "site_5112",
    "site_5119", "site_5106", "site_5115",
]

OUTPUT = Path("data/raw/xkdr/2024")
OUTPUT.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "station_id", "parameter_name", "unit",
    "collected_at", "value"
}


def fetch_month(session, month):
    start = date(2024, month, 1)
    end = date(2024, month, monthrange(2024, month)[1])

    params = [
        *(("station", station) for station in STATIONS),
        ("parameter", "PM2.5"),
        ("start", start.isoformat()),
        ("end", end.isoformat()),
        ("agg", "raw"),
        ("format", "csv"),
    ]

    response = None

    for attempt in range(4):
        response = session.get(
            BASE_URL, params=params, timeout=(15, 120)
        )

        if response.status_code not in (429, 500, 502, 503, 504):
            break

        if attempt < 3:
            wait = min(2 ** (attempt + 1), 16)
            print(f"Temporary HTTP {response.status_code}; retrying in {wait}s")
            time.sleep(wait)

    response.raise_for_status()

    raw_bytes = response.content
    text = raw_bytes.decode("utf-8-sig")
    filename = f"2024-{month:02d}"

    safe_headers = {
        key: value for key, value in response.headers.items()
        if key.lower().startswith("x-")
        or key.lower() in ("content-type", "content-length")
    }

    truncated = response.headers.get("X-Truncated", "").lower()
    row_header = response.headers.get("X-Row-Count")

    metadata = {
        "month": filename,
        "status_code": response.status_code,
        "request_url": response.url,
        "response_headers": safe_headers,
        "bytes": len(raw_bytes),
        "truncated": truncated,
        "api_row_count": row_header,
    }

    (OUTPUT / f"{filename}.http.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if truncated != "false":
        rejected = OUTPUT / f"{filename}.rejected.csv"
        rejected.write_bytes(raw_bytes)
        raise RuntimeError(
            f"{filename}: truncation header was {truncated!r}. "
            f"Response preserved at {rejected}"
        )

    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        raise RuntimeError(f"{filename}: CSV has no header")

    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise RuntimeError(
            f"{filename}: missing columns {sorted(missing)}"
        )

    rows = list(reader)

    if row_header is not None and int(row_header) != len(rows):
        raise RuntimeError(
            f"{filename}: API reports {row_header} rows, "
            f"CSV contains {len(rows)}"
        )

    unexpected_stations = {
        row["station_id"] for row in rows
        if row["station_id"] not in STATIONS
    }
    unexpected_parameters = {
        row["parameter_name"] for row in rows
        if row["parameter_name"] != "PM2.5"
    }

    if unexpected_stations or unexpected_parameters:
        raise RuntimeError(
            f"{filename}: unexpected station or pollutant"
        )

    (OUTPUT / f"{filename}.csv").write_bytes(raw_bytes)

    print(
        f"{filename}: HTTP {response.status_code} | "
        f"{len(rows):,} rows | {len(raw_bytes):,} bytes"
    )

    return {
        "month": filename,
        "rows": len(rows),
        "bytes": len(raw_bytes),
        "file": f"{filename}.csv",
    }


def main():
    api_key = os.environ.get("XKDR_API_KEY")

    if not api_key:
        raise RuntimeError(
            "XKDR_API_KEY is not configured in this PowerShell session."
        )

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/csv",
        "User-Agent": "AIRGUARD-research/0.1",
    })

    results = []

    for month in range(1, 13):
        print(f"\nDownloading month {month}/12...")
        results.append(fetch_month(session, month))
        time.sleep(0.3)

    manifest = {
        "dataset": "XKDR India Air Quality Database",
        "year": 2024,
        "city": "Mumbai",
        "parameter": "PM2.5",
        "aggregation": "raw",
        "stations": STATIONS,
        "timezone": "IST (UTC+05:30)",
        "monthly_results": results,
        "total_rows": sum(x["rows"] for x in results),
        "total_bytes": sum(x["bytes"] for x in results),
        "note": "Original monthly CSV response bytes preserved.",
    }

    path = OUTPUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDOWNLOAD COMPLETE")
    print(f"Total rows: {manifest['total_rows']:,}")
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
