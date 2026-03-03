"""
VHR download + crop in EPSG:3035, with:
1) strict 1280x1280 px check (count how many fail)
2) if multiple TIFFs are present, mosaic them on-the-fly per window to fill coverage

Notes:
- We keep everything in the VHR raster CRS (typically EPSG:3035) and write output in that CRS.
- Mosaic is done only for the subset of tifs that intersect the window (fast enough, avoids building a full mosaic).
- We still keep the previous “no invalid pixels inside polygon” check.
  If you want to relax that later, it’s a single if-block.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import geopandas as gpd
from shapely import wkt
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.io import MemoryFile

from cdse_auth import get_access_token
from helpers import odata_get_with_retry, log_line, load_done_set, mark_done

load_dotenv()

# -----------------------
# CONFIG
# -----------------------
META_PARQUET = "parquet/windows_best_s2_with_vza_angleDiffLE5_withSCL_split_byDatastrip.parquet"
CATALOG_GPKG = "gpkg/vhr2024_joined_acq_catalog.gpkg"

OUT_ROOT = Path("data")
TMP_ROOT = Path("tmp_downloads")
TMP_ROOT.mkdir(exist_ok=True)

N_DATASTRIPS = None  # set None or remove slicing for full run

WIN_CRS = "EPSG:3035"

DATASET_NAME = "VHR_IMAGE_2024"

CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

OUT_KEPT_PARQUET = "parquet/windows_with_downloaded_vhr.parquet"

# Expected output size in pixels (2m * 1280 = 2560m)
EXPECTED_PX = 1280

# -----------------------
# Name logic: SAFE.zip -> COG name
# -----------------------
def safezip_to_cog_name(product_name: str) -> str:
    name = re.sub(r"\.zip$", "", product_name, flags=re.IGNORECASE)
    name = re.sub(r"\.SAFE$", "", name, flags=re.IGNORECASE)
    return f"{name}_COG"

def sanitize_filename(name: str) -> str:
    return re.sub(r"[<>:\"/\\|?*]", "_", name)

# -----------------------
# Catalogue resolve: exact Name -> UUID
# -----------------------
def resolve_uuid_by_exact_name(cog_name: str, token: str, refresh_token_fn) -> tuple[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    name_escaped = cog_name.replace("'", "''")

    dataset_filter = (
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'datasetFull' and "
        f"att/OData.CSC.StringAttribute/Value eq '{DATASET_NAME}'"
        ")"
    )
    name_filter = f"Name eq '{name_escaped}'"
    odata_filter = " and ".join([dataset_filter, name_filter])

    params = {"$filter": odata_filter, "$select": "Id,Name", "$top": 2}
    js, headers2 = odata_get_with_retry(os.getenv("CATALOGUE_URL"), headers, params, refresh_token_fn)

    vals = js.get("value", [])
    if not vals:
        raise RuntimeError(f"COG product not found by exact Name: {cog_name}")

    uuid_id = str(vals[0]["Id"])
    token2 = headers2["Authorization"].split(" ", 1)[1]
    return uuid_id, token2

# -----------------------
# Download + extract GeoTIFF(s)
# -----------------------
def download_and_extract_tifs(uuid_id: str, cog_name: str, token: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{os.getenv('DOWNLOAD_URL')}({uuid_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    safe_name = sanitize_filename(cog_name)
    zip_path = out_dir / f"{safe_name}.zip"

    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        if resp.status_code != 200:
            print(f"[ERROR] Download failed status={resp.status_code} url={url}")
            print(resp.text[:1200])
        resp.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tif_paths: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            fn = member.filename.lower()
            if fn.endswith((".tif", ".tiff")):
                tif_out = out_dir / os.path.basename(member.filename)
                with zf.open(member) as src, open(tif_out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                tif_paths.append(tif_out)

    try:
        zip_path.unlink()
    except Exception:
        pass

    if not tif_paths:
        raise RuntimeError(f"No GeoTIFFs found in ZIP for '{cog_name}' (uuid={uuid_id})")

    return tif_paths

# -----------------------
# Crop + (optional) mosaic per window + strict 1280x1280 check
# -----------------------
def crop_vhr_windows_by_geometry_filter(
    tif_paths: list[Path],
    windows_df: pd.DataFrame,
    out_dir: Path,
    expected_px: int = 1280,
    max_mosaic_pixels: int = 250_000_000,  # safety: ~250M px total (bands excluded)
):
    """
    Optimized: build ONE mosaic per datastrip (limited to bbox of all windows),
    then crop all windows from that mosaic.

    Safety:
      - If the limited mosaic would still be huge, fallback to per-window mosaic.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dsets = [rasterio.open(str(p)) for p in tif_paths]

    kept: list[dict] = []
    skipped: list[tuple[str, str]] = []

    counts = {
        "windows": 0,
        "no_intersect": 0,
        "empty": 0,
        "mask_error": 0,
        "wrong_shape": 0,
        "mosaic_once_used": 0,
        "mosaic_per_window_fallback": 0,
        "saved": 0,
    }

    # --- prepare windows geometries once ---
    gdf = windows_df.copy()
    gdf["geometry"] = gdf["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=WIN_CRS)

    try:
        # Pick a reference CRS/profile from the first dataset
        ref = dsets[0]
        ref_crs = ref.crs

        # Reproject all window geometries into raster CRS (ref_crs)
        geom_in_ref = gdf.geometry.to_crs(ref_crs)

        # Compute bbox covering all windows (in raster CRS)
        minx, miny, maxx, maxy = geom_in_ref.total_bounds
        bounds = (float(minx), float(miny), float(maxx), float(maxy))

        # Estimate mosaic size in pixels (rough) to decide if we do it once
        # Use ref resolution
        resx = abs(ref.transform.a)
        resy = abs(ref.transform.e)
        est_w = int(np.ceil((bounds[2] - bounds[0]) / resx))
        est_h = int(np.ceil((bounds[3] - bounds[1]) / resy))
        est_pixels = est_w * est_h

        use_mosaic_once = est_pixels <= max_mosaic_pixels and est_w > 0 and est_h > 0

        mosaic_ds = None
        memfile = None

        if use_mosaic_once:
            counts["mosaic_once_used"] += 1

            # Merge only within the windows bbox (fast + smaller)
            mosaic_arr, mosaic_transform = merge(
                dsets,
                bounds=bounds,
                # keep native resolution (if you want force: res=(resx, resy))
                nodata=ref.nodata,
            )

            profile = ref.profile.copy()
            profile.update(
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                transform=mosaic_transform,
                driver="GTiff",
            )

            memfile = MemoryFile()
            mosaic_ds = memfile.open(**profile)
            mosaic_ds.write(mosaic_arr)

        # --- per-window processing ---
        for idx, row in enumerate(tqdm(gdf.itertuples(index=False), total=len(gdf), desc="Crop VHR")):
            counts["windows"] += 1

            win_id = row.window_id
            out_path = out_dir / f"{win_id}.tif"

            if out_path.exists():
                kept.append({"window_id": win_id, "vhr_path": str(out_path)})
                continue

            # geometry in raster CRS
            g_ref = geom_in_ref.iloc[idx]

            # Decide source dataset for crop:
            # - If we have a global mosaic, use it
            # - else fallback to per-window mosaic of intersecting tiles
            src = mosaic_ds
            close_src = False
            close_mem = False

            if src is None:
                # fallback: find intersecting tiles
                intersecting = []
                for ds in dsets:
                    gb = g_ref.bounds
                    b = ds.bounds
                    if not (gb[2] <= b.left or gb[0] >= b.right or gb[3] <= b.bottom or gb[1] >= b.top):
                        intersecting.append(ds)

                if not intersecting:
                    counts["no_intersect"] += 1
                    skipped.append((win_id, "no_intersect"))
                    continue

                if len(intersecting) == 1:
                    src = intersecting[0]
                else:
                    counts["mosaic_per_window_fallback"] += 1
                    arr, tr = merge(intersecting, nodata=ref.nodata)
                    prof = intersecting[0].profile.copy()
                    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr, driver="GTiff")

                    mf = MemoryFile()
                    tmp = mf.open(**prof)
                    tmp.write(arr)

                    src = tmp
                    close_src = True
                    close_mem = True
                    tmp_memfile = mf  # keep reference for closing

            try:
                out_img, out_transform = mask(
                    src,
                    [g_ref.__geo_interface__],
                    crop=True,
                    filled=False,
                )
            except ValueError as e:
                counts["mask_error"] += 1
                skipped.append((win_id, f"mask_error:{e}"))
                if close_src:
                    src.close()
                if close_mem:
                    tmp_memfile.close()
                continue

            if close_src:
                src.close()
            if close_mem:
                tmp_memfile.close()

            if out_img.size == 0 or out_img.shape[1] == 0 or out_img.shape[2] == 0:
                counts["empty"] += 1
                skipped.append((win_id, "empty"))
                continue

            H, W = out_img.shape[1], out_img.shape[2]

            # strict pixel size check
            if H != expected_px or W != expected_px:
                counts["wrong_shape"] += 1
                skipped.append((win_id, f"wrong_shape={H}x{W}"))
                continue

            nodata = ref.nodata if ref.nodata is not None else 0
            data_to_write = out_img.filled(nodata)

            dtype_kind = np.dtype(data_to_write.dtype).kind
            predictor = 3 if dtype_kind == "f" else 2

            profile = ref.profile.copy()
            profile.update(
                height=H,
                width=W,
                transform=out_transform,
                driver="GTiff",
                tiled=True,
                compress="deflate",
                predictor=predictor,
                nodata=nodata,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data_to_write)

                # preserve scale/offset/tags
                try:
                    dst.scales = ref.scales
                    dst.offsets = ref.offsets
                except Exception:
                    pass
                try:
                    dst.update_tags(**ref.tags())
                    for b in range(1, ref.count + 1):
                        dst.update_tags(b, **ref.tags(b))
                except Exception:
                    pass

            kept.append({"window_id": win_id, "vhr_path": str(out_path)})
            counts["saved"] += 1

        # close mosaic
        if mosaic_ds is not None:
            mosaic_ds.close()
        if memfile is not None:
            memfile.close()

    finally:
        for ds in dsets:
            ds.close()

    return kept, skipped, counts

# -----------------------
# Map datastrip -> SAFE product_name from GPKG
# -----------------------
CAT_DS_COL = "datastrip"
CAT_PRODUCT_NAME_COL = "product_name"

def build_datastrip_to_product_name_map(gpkg_path: str) -> dict[str, str]:
    cat = gpd.read_file(gpkg_path)

    if CAT_DS_COL not in cat.columns or CAT_PRODUCT_NAME_COL not in cat.columns:
        raise RuntimeError(
            f"Expected columns not found in {gpkg_path}\n"
            f"Need: {CAT_DS_COL}, {CAT_PRODUCT_NAME_COL}\n"
            f"Columns are: {list(cat.columns)}"
        )

    sub = cat[[CAT_DS_COL, CAT_PRODUCT_NAME_COL]].dropna().drop_duplicates()
    m = {}
    for ds, pn in zip(sub[CAT_DS_COL].astype(str), sub[CAT_PRODUCT_NAME_COL].astype(str)):
        if ds not in m:
            m[ds] = pn
    return m

# -----------------------
# MAIN
# -----------------------
def main():
    if not CDSE_USERNAME or not CDSE_PASSWORD:
        raise RuntimeError("Please set env vars CDSE_USERNAME and CDSE_PASSWORD.")

    for split in ["train", "val", "test"]:
        (OUT_ROOT / split / "vhr").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "s2").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(META_PARQUET)

    required_cols = {"datastrip", "split", "window_id", "geometry_wkt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {META_PARQUET}: {missing}")

    # leakage check
    bad = df.groupby("datastrip")["split"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise RuntimeError(f"Some datastrips appear in multiple splits:\n{bad.head(20)}")

    datastrips = sorted(df["datastrip"].unique())
    if N_DATASTRIPS is not None:
        datastrips = datastrips[:N_DATASTRIPS]

    done = load_done_set()
    log_line(f"Already done datastrips: {len(done)}")

    token = get_access_token(CDSE_USERNAME, CDSE_PASSWORD)

    ds_to_safe_name = build_datastrip_to_product_name_map(CATALOG_GPKG)
    print(f"Loaded datastrip->product_name map: {len(ds_to_safe_name)} entries")

    kept_rows_all = []
    per_ds_summary = []

    total_count = len(datastrips)

    for idx, ds in enumerate(datastrips, start=1):
        if ds in done:
            continue

        ds_rows = df[df["datastrip"] == ds].copy()
        split = ds_rows["split"].iloc[0]
        nwin = len(ds_rows)

        print(f"\n=== Datastrip {idx}/{total_count} [{ds}] | split={split} | windows={nwin} ===")

        if ds not in ds_to_safe_name:
            print(f"[WARN] datastrip not found in catalog gpkg: {ds}")
            per_ds_summary.append(
                {"datastrip": ds, "split": split, "windows": nwin, "kept": 0, "skipped": nwin, "error": "no_product_name"}
            )
            continue

        safe_name = ds_to_safe_name[ds]
        cog_name = safezip_to_cog_name(safe_name)
        print("SAFE:", safe_name)
        print("COG :", cog_name)

        ds_tmp_dir = TMP_ROOT / sanitize_filename(ds)
        if ds_tmp_dir.exists():
            shutil.rmtree(ds_tmp_dir, ignore_errors=True)
        ds_tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            uuid_id, token = resolve_uuid_by_exact_name(
                cog_name,
                token,
                refresh_token_fn=lambda: get_access_token(CDSE_USERNAME, CDSE_PASSWORD),
            )
            log_line(f"UUID: {uuid_id}")

            tif_paths = download_and_extract_tifs(uuid_id, cog_name, token, ds_tmp_dir)
            print(f"Extracted {len(tif_paths)} GeoTIFF(s)")

            out_vhr_dir = OUT_ROOT / split / "vhr"

            kept, skipped, counts = crop_vhr_windows_by_geometry_filter(
                tif_paths=tif_paths,
                windows_df=ds_rows,
                out_dir=out_vhr_dir,
                expected_px=EXPECTED_PX,
            )

            print(
                f"Kept windows: {len(kept)} / {nwin} | "
                f"wrong_shape={counts['wrong_shape']} | "
                f"mosaic_once_used={counts['mosaic_once_used']} | "
                f"mosaic_per_window_fallback={counts['mosaic_per_window_fallback']} | "
                f"no_intersect={counts['no_intersect']}"
            )
            if skipped:
                print("Skipped (first 30):", skipped[:30])

            if kept:
                kept_ids = {k["window_id"] for k in kept}
                kept_df = ds_rows[ds_rows["window_id"].isin(kept_ids)].copy()
                kept_meta = pd.DataFrame(kept)
                kept_df = kept_df.merge(kept_meta, on="window_id", how="left")
                kept_rows_all.append(kept_df)

            per_ds_summary.append(
                {
                    "datastrip": ds,
                    "split": split,
                    "windows": nwin,
                    "kept": int(len(kept)),
                    "skipped": int(nwin - len(kept)),
                    "wrong_shape": int(counts["wrong_shape"]),
                    "mosaic_once_used": int(counts["mosaic_once_used"]),
                    "mosaic_per_window_fallback": int(counts["mosaic_per_window_fallback"]),
                    "no_intersect": int(counts["no_intersect"]),
                    "error": "",
                }
            )

            log_line(
                f"OK datastrip={ds} kept={len(kept)} "
                f"wrong_shape={counts['wrong_shape']} "
                f"mosaic_once_used={counts['mosaic_once_used']} "
                f"mosaic_per_window_fallback={counts['mosaic_per_window_fallback']}"
            )
            mark_done(ds)

        except Exception as e:
            log_line(f"[ERROR] datastrip={ds} err={e}")
            per_ds_summary.append(
                {"datastrip": ds, "split": split, "windows": nwin, "kept": 0, "skipped": nwin, "error": str(e)}
            )

        finally:
            shutil.rmtree(ds_tmp_dir, ignore_errors=True)

    # Save parquet with only successfully-kept windows
    if kept_rows_all:
        kept_all = pd.concat(kept_rows_all, ignore_index=True)
    else:
        kept_all = df.iloc[0:0].copy()

    kept_all.to_parquet(OUT_KEPT_PARQUET, index=False)
    print(f"\nSaved kept windows parquet: {OUT_KEPT_PARQUET}  rows={len(kept_all)}")

    # Save summary CSV
    summary_df = pd.DataFrame(per_ds_summary)
    summary_df.to_csv("vhr_download_summary.csv", index=False)
    print("Saved summary: vhr_download_summary.csv")

if __name__ == "__main__":
    main()
