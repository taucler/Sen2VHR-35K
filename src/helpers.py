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

load_dotenv()

SESSION = requests.Session()

to_4326 = Transformer.from_crs(3035, 4326, always_xy=True)

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