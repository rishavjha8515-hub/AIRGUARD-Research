#!/usr/bin/env python3
"""Fetch and preserve XKDR metadata responses; does not download/modify observations."""
import argparse
import json
import os
from pathlib import Path
import sys
import requests

BASE = "https://airquality.xkdr.org"
DEMO_KEY = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Mumbai")
    parser.add_argument("--out", default="data/raw/xkdr_metadata")
    args = parser.parse_args()

    key = os.environ.get("XKDR_API_KEY", DEMO_KEY)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {key}"}

    endpoints = {
        "meta": ("/v1/meta", {}),
        "stations": ("/v1/stations", {"city": args.city}),
        "parameters": ("/v1/parameters", {}),
    }

    failures = []
    for name, (path, params) in endpoints.items():
        url = BASE + path
        try:
            response = requests.get(url, headers=headers, params=params, timeout=45)
            # Preserve response body exactly as received, even on non-200 responses.
            (out / f"{name}.response.txt").write_bytes(response.content)
            info = {
                "url": response.url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "row_count_header": response.headers.get("X-Row-Count"),
                "truncated_header": response.headers.get("X-Truncated"),
                "bytes": len(response.content),
            }
            (out / f"{name}.http.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8"
            )
            print(f"{name}: HTTP {response.status_code}, {len(response.content)} bytes")
            if not response.ok:
                failures.append(name)
        except requests.RequestException as exc:
            print(f"{name}: REQUEST FAILED: {exc}")
            failures.append(name)

    print(f"\nRaw metadata responses saved to: {out.resolve()}")
    if failures:
        print("Some requests failed. Inspect the saved HTTP metadata and response text.")
        sys.exit(1)
    print("No observations were downloaded or transformed.")

if __name__ == "__main__":
    main()
