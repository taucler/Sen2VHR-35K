import requests
from datetime import timedelta
from shapely.ops import transform
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
import time
import random
import os
from dotenv import load_dotenv
from datetime import datetime, UTC
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

load_dotenv()

SESSION = requests.Session()
_token = None
_token_exp = 0

to_4326 = Transformer.from_crs(3035, 4326, always_xy=True)

# -----------------------
# Grid helpers
# -----------------------

def get_datastrip_footprints(cand_parquet: str, catalog_gpkg: str, crs: str) -> gpd.GeoDataFrame:
    cand = pd.read_parquet(cand_parquet)
    if "datastrip" not in cand.columns:
        raise RuntimeError(f"'datastrip' column not found in {cand_parquet}")

    ds_list = sorted(cand["datastrip"].dropna().unique().tolist())

    cat = gpd.read_file(catalog_gpkg)
    if "datastrip" not in cat.columns:
        raise RuntimeError(f"'datastrip' column not found in {catalog_gpkg}. Columns: {list(cat.columns)}")

    if cat.crs is None:
        cat = cat.set_crs("EPSG:4326")

    cat = cat[cat["datastrip"].isin(ds_list)].copy()
    cat = cat.to_crs(crs)

    # ---- FIX INVALID GEOMETRIES ----
    # Shapely 2.x: make_valid exists; buffer(0) is a common fallback
    try:
        cat["geometry"] = cat["geometry"].make_valid()
    except Exception:
        cat["geometry"] = cat["geometry"].buffer(0)

    # drop empties after repair
    cat = cat[~cat.geometry.is_empty & cat.geometry.notna()].copy()

    # dissolve
    cat = cat.dissolve(by="datastrip", as_index=False)

    return cat[["datastrip", "geometry"]].copy()

def estimate_utm_epsg(geom_wgs84, crs) -> int:
    gs = gpd.GeoSeries([geom_wgs84], crs=crs)
    utm = gs.estimate_utm_crs()
    epsg = utm.to_epsg()
    if epsg is None:
        raise RuntimeError("Could not estimate UTM EPSG.")
    return int(epsg)


def generate_windows_max_count(geom_utm, window_m: float, edge_buffer_m: float, try_four_offsets: bool):
    """
    Windows are axis-aligned squares in UTM, touching inside the datastrip.
    We apply an inward buffer to the datastrip footprint to avoid boundary overlap issues.
    We try up to 4 grid origins and return the one with the maximum count.
    """
    inner = geom_utm.buffer(-edge_buffer_m)
    if inner.is_empty:
        return [], (0.0, 0.0)

    minx, miny, maxx, maxy = inner.bounds
    if (maxx - minx) < window_m or (maxy - miny) < window_m:
        return [], (0.0, 0.0)

    stride = window_m  # touching windows

    offsets = [(0.0, 0.0)]
    if try_four_offsets:
        half = window_m / 2.0
        offsets = [(0.0, 0.0), (half, 0.0), (0.0, half), (half, half)]

    best_wins = []
    best_offset = (0.0, 0.0)

    # tiny shrink to avoid floating boundary issues
    inner2 = inner.buffer(-0.01)

    for ox, oy in offsets:
        xs = np.arange(minx + ox, maxx - window_m + 1e-9, stride)
        ys = np.arange(miny + oy, maxy - window_m + 1e-9, stride)

        wins = []
        for x in xs:
            for y in ys:
                w = box(x, y, x + window_m, y + window_m)
                if w.within(inner2):
                    wins.append(w)

        if len(wins) > len(best_wins):
            best_wins = wins
            best_offset = (ox, oy)

    return best_wins, best_offset

def bbox_wkt_4326(geom_3035):
    geom_4326 = transform(to_4326.transform, geom_3035)
    minx, miny, maxx, maxy = geom_4326.bounds
    return box(minx, miny, maxx, maxy).wkt

def get_json_with_retry(url, headers=None, max_tries=8, base_sleep=1.0, timeout=60):
    last = None
    for attempt in range(1, max_tries + 1):
        r = SESSION.get(url, headers=headers, timeout=timeout)
        last = r

        if r.status_code == 200:
            return r.json()

        # Treat these as transient (rate limit / WAF / gateway)
        if r.status_code in (403, 429, 500, 502, 503, 504):
            # If server provides Retry-After, respect it
            ra = r.headers.get("Retry-After")
            if ra:
                sleep_s = float(ra)
            else:
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.7)

            print(f"[HTTP {r.status_code}] attempt {attempt}/{max_tries} → sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue

        # Non-transient → show body and fail fast
        print("Non-retryable error body (first 1000 chars):")
        print(r.text[:1000])
        r.raise_for_status()

    # exhausted retries
    print("Exhausted retries. Last error body (first 1000 chars):")
    print((last.text[:1000] if last is not None else "no response"))
    last.raise_for_status()

def query_s2_candidates(token: str, aoi_wkt_3035: BaseGeometry, t0_utc, hours=12, top=100):
    """
    Returns list of S2 product dicts (metadata only).
    t0_utc: pandas.Timestamp or datetime (UTC)
    """
    headers = None # {"Authorization": f"Bearer {token}"}

    t_min = (t0_utc - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    t_max = (t0_utc + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    aoi_wkt_4326 = bbox_wkt_4326(aoi_wkt_3035)

    # NOTE: Collection/Name filter works for S2 catalogue queries
    # If your deployment uses a different collection name, you can drop it and filter by Name prefix instead.
    odata_filter = (
        "Collection/Name eq 'SENTINEL-2' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt_4326}') and "
        f"ContentDate/Start ge {t_min} and "
        f"ContentDate/Start le {t_max}"
    )

    url = (
        f"{os.getenv("CATALOGUE_URL")}?"
        f"$filter={odata_filter}"
        f"&$select=Id,Name,ContentDate,GeoFootprint"
        f"&$top={top}"
        f"&$orderby=ContentDate/Start"
        # optional, but useful:
        # f"&$expand=Attributes"
    )

    data = get_json_with_retry(url, headers=headers)
    return data.get("value", [])

def request_with_retry(payload, token, max_tries=4):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_resp = None
    for attempt in range(max_tries):
        resp = requests.post(os.getenv("STATS_URL"), headers=headers, json=payload, timeout=180)
        last_resp = resp

        # Token expired / not authorized -> caller should refresh token
        if resp.status_code in (401, 403):
            return resp

        # Too many requests / transient server problems -> backoff and retry
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_s = min(2 ** attempt, 10)
            time.sleep(sleep_s)
            continue

        return resp

    return last_resp

def odata_get_with_retry(url: str, headers: dict, params: dict, refresh_token_fn, max_tries=6):
    last = None
    for attempt in range(1, max_tries + 1):
        r = requests.get(url, headers=headers, params=params, timeout=60)

        if r.status_code == 200:
            return r.json(), headers

        if r.status_code in (401, 403):
            # refresh token and retry
            try:
                tok = refresh_token_fn()
                headers = dict(headers)
                headers["Authorization"] = f"Bearer {tok}"
            except Exception as e:
                last = e

        if r.status_code in (403, 429, 500, 502, 503, 504):
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(min(60, (2 ** (attempt - 1)) + random.random()))
            continue

        r.raise_for_status()

    raise RuntimeError(f"OData GET failed after {max_tries} tries: {last}")

def get_sh_token():
    global _token, _token_exp
    now = time.time()
    # refresh ~60s early
    if _token and now < (_token_exp - 60):
        return _token

    r = SESSION.post(
        os.getenv("TOKEN_URL"),
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("CLIENT_ID"),
            "client_secret": os.getenv("CLIENT_SECRET"),
        },
        timeout=60,
    )
    r.raise_for_status()
    js = r.json()
    _token = js["access_token"]
    # expires_in is usually present (seconds)
    _token_exp = now + float(js.get("expires_in", 600))
    return _token


# -----------------------
# Logging utils
# -----------------------

LOG_PATH = Path("vhr_download.log")
DONE_PATH = Path("vhr_done_datastrips.txt")

def log_line(msg: str):
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def mark_done(datastrip: str):
    with open(DONE_PATH, "a", encoding="utf-8") as f:
        f.write(datastrip + "\n")

def load_done_set() -> set[str]:
    if not DONE_PATH.exists():
        return set()
    with open(DONE_PATH, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}