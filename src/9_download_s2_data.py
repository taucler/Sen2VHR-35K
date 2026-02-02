from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Iterable
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from shapely import wkt

load_dotenv()

# -----------------------
# CONFIG
# -----------------------
IN_PARQUET = "parquet/windows_with_downloaded_vhr.parquet"

OUT_ROOT = Path("data")

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Geometry CRS
WIN_CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/3035"

# Download config
BANDS_10M = ["B04", "B03", "B02", "B08"]  # Blue, Green, Red, NIR
RES_METERS = 10  # 10 m bands

# Request / retry
TIMEOUT_S = 180
MAX_RETRIES = 5
SLEEP_BASE = 0.1

# -----------------------
# AUTH (client credentials)
# -----------------------
def get_token() -> str:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.")

    r = requests.post(
        os.getenv("TOKEN_URL"),
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# -----------------------
# Sentinel Hub evalscript
# -----------------------
def make_evalscript(bands: list[str]) -> str:
    # Output as UINT16 reflectance-like DNs (typically 0..10000 for S2 L2A)
    # Include dataMask as last band if you want later QC; default: no.
    in_bands = ", ".join([f'"{b}"' for b in bands])
    out_expr = ", ".join([f"s.{b}" for b in bands])

    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: [{in_bands}],
      units: "DN"
    }}],
    output: {{
      id: "default",
      bands: {len(bands)},
      sampleType: "UINT16"
    }}
  }};
}}

function evaluatePixel(s) {{
  return [{out_expr}];
}}
"""


# -----------------------
# Helpers
# -----------------------
def iso_z(ts: pd.Timestamp) -> str:
    # ensure Z format
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_output_grid_from_geom(geom_3035, res_m: float) -> tuple[list[float], int, int]:
    """
    Returns (bbox, width, height) for Process API.
    bbox is [minx, miny, maxx, maxy] in EPSG:3035.
    width/height computed from bbox and resolution.
    """
    minx, miny, maxx, maxy = geom_3035.bounds
    w_m = maxx - minx
    h_m = maxy - miny

    # Avoid zero or negative due to invalid geom
    if w_m <= 0 or h_m <= 0:
        raise ValueError("Invalid geometry bounds")

    width = int(math.ceil(w_m / res_m))
    height = int(math.ceil(h_m / res_m))

    # safety clamp
    width = max(1, width)
    height = max(1, height)

    bbox = [float(minx), float(miny), float(maxx), float(maxy)]
    return bbox, width, height


def request_with_retries(session: requests.Session, url: str, headers: dict, json_payload: dict) -> bytes:
    """
    POST with retries. Returns response content (bytes) on success.
    Handles 401 (refresh token externally), 429, 5xx.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, headers=headers, json=json_payload, timeout=TIMEOUT_S)
            if r.status_code == 200:
                return r.content

            # Rate limit / transient server issues
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                sleep_s = SLEEP_BASE * (2 ** (attempt - 1))
                time.sleep(min(60, sleep_s))
                continue

            # Other hard errors
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")

        except Exception as e:
            last_err = e
            sleep_s = SLEEP_BASE * (2 ** (attempt - 1))
            time.sleep(min(60, sleep_s))

    raise RuntimeError(f"Request failed after {MAX_RETRIES} retries: {last_err}")


# -----------------------
# Main download per window
# -----------------------
def download_one_window(
    session: requests.Session,
    token: str,
    row,
    out_path: Path,
    evalscript: str,
    res_m: float,
) -> None:
    geom_3035 = wkt.loads(row.geometry_wkt)
    bbox, width, height = compute_output_grid_from_geom(geom_3035, res_m)

    # Time range: day containing s2_time (plus optional pad)
    t = pd.to_datetime(row.s2_time, utc=True)
    day_start = t.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "geometry": geom_3035.__geo_interface__,  # polygon clip
                "properties": {"crs": WIN_CRS_URI},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": iso_z(day_start), "to": iso_z(day_end)},
                        "mosaickingOrder": "mostRecent",
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }

    tif_bytes = request_with_retries(session, os.getenv("PROCESS_URL"), headers, payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tif_bytes)


def main():
    df = pd.read_parquet(IN_PARQUET)

    required = {"window_id", "split", "geometry_wkt", "s2_time"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {IN_PARQUET}: {missing}")

    # Ensure output dirs exist
    for split in ["train", "val", "test"]:
        (OUT_ROOT / split / "s2").mkdir(parents=True, exist_ok=True)

    evalscript = make_evalscript(BANDS_10M)

    token = get_token()
    session = requests.Session()

    ok = 0
    err = 0
    errors = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Download S2"):
        out_path = OUT_ROOT / row.split / "s2" / f"{row.window_id}.tif"
        if out_path.exists():
            continue

        try:
            download_one_window(session, token, row, out_path, evalscript, RES_METERS)
            ok += 1

        except Exception as e:
            # If token expired, refresh once and retry immediately
            msg = str(e)
            if "401" in msg or "invalid_token" in msg.lower():
                try:
                    token = get_token()
                    download_one_window(session, token, row, out_path, evalscript, RES_METERS)
                    ok += 1
                    continue
                except Exception as e2:
                    e = e2

            err += 1
            errors.append({"window_id": row.window_id, "split": row.split, "error": str(e)[:2000]})

    # Save error log for post-mortem
    if errors:
        pd.DataFrame(errors).to_csv("s2_download_errors.csv", index=False)

    print(f"\nDone. OK={ok}  ERR={err}")
    if errors:
        print("Wrote: s2_download_errors.csv")


if __name__ == "__main__":
    main()
