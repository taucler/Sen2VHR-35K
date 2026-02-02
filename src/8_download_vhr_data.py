"""
Script: Download and Process VHR Data

This script automates the process of downloading Very High Resolution (VHR) satellite imagery data from the Copernicus Data Space Ecosystem (CDSE). It performs the following tasks:

1. **Authentication**: Retrieves an access token for the CDSE API using user credentials.
2. **Catalog Query**: Resolves datastrip and product names to unique identifiers (UUIDs) for downloading.
3. **Data Download**: Downloads VHR data in SAFE format and extracts GeoTIFF files.
4. **Window Cropping**: Crops the downloaded VHR imagery to match predefined training windows, ensuring full coverage within the window geometry.
5. **Data Filtering**: Filters out incomplete or invalid windows based on geometry and data quality.
6. **Output Management**: Saves the processed VHR data and metadata to specified output directories and logs the process.

Purpose: This script is part of the VHR super-resolution pipeline. It ensures that the VHR data is properly downloaded, processed, and aligned with the training windows for further analysis and model training.

Dependencies:
- Python libraries: `os`, `re`, `shutil`, `zipfile`, `pathlib`, `random`, `time`, `datetime`, `numpy`, `pandas`, `requests`, `tqdm`, `geopandas`, `shapely`, `rasterio`.
- External tools: Copernicus Data Space Ecosystem API.

Usage:
Run the script as a standalone program. Ensure that the required input files and credentials are correctly configured in the `CONFIG` section.

"""

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

from cdse_auth import get_access_token
from helpers import odata_get_with_retry, log_line, load_done_set, mark_done

load_dotenv()

# -----------------------
# CONFIG
# -----------------------
META_PARQUET = "parquet/windows_best_s2_with_vza_angleDiffLE4_withSCL_split_byDatastrip.parquet"
CATALOG_GPKG = "gpkg/vhr2024_joined_acq_catalog.gpkg"

OUT_ROOT = Path("data")
TMP_ROOT = Path("tmp_downloads")
TMP_ROOT.mkdir(exist_ok=True)

N_DATASTRIPS = None  # set None or remove slicing for full run

WIN_CRS = "EPSG:3035"

DATASET_NAME = "VHR_IMAGE_2024" # exists with 2021, 2018 and 2015 versions too, hasn't been tested 

CDSE_USERNAME = os.getenv("CDSE_USERNAME")  
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

# output parquet with only kept windows
OUT_KEPT_PARQUET = "parquet/windows_with_downloaded_vhr.parquet"

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

    url = f"{os.getenv("DOWNLOAD_URL")}({uuid_id})/$value"
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

    tif_paths = []
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
# Crop windows by polygon geometry, filter partials, and return kept window_ids + valid_frac
# -----------------------
def crop_vhr_windows_by_geometry_filter(tif_paths, windows_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    dsets = [rasterio.open(p) for p in tif_paths]
    kept = []
    skipped = []

    try:
        for row in tqdm(windows_df.itertuples(index=False), total=len(windows_df), desc="Crop VHR"):
            win_id = row.window_id
            out_path = out_dir / f"{win_id}.tif"

            geom_3035 = wkt.loads(row.geometry_wkt)

            chosen_ds, geom_in_ds = None, None
            for ds in dsets:
                g = gpd.GeoSeries([geom_3035], crs=WIN_CRS).to_crs(ds.crs).iloc[0]
                gb = g.bounds
                b = ds.bounds
                if not (gb[2] <= b.left or gb[0] >= b.right or gb[3] <= b.bottom or gb[1] >= b.top):
                    chosen_ds, geom_in_ds = ds, g
                    break

            if chosen_ds is None:
                skipped.append((win_id, "no_intersect"))
                continue

            try:
                out_img, out_transform = mask(
                    chosen_ds,
                    [geom_in_ds.__geo_interface__],
                    crop=True,
                    filled=False
                )
            except ValueError as e:
                skipped.append((win_id, f"mask_error:{e}"))
                continue

            if out_img.size == 0 or out_img.shape[1] == 0 or out_img.shape[2] == 0:
                skipped.append((win_id, "empty"))
                continue

            H, W = out_img.shape[1], out_img.shape[2]

            # True outside polygon
            outside_poly = geometry_mask(
                [geom_in_ds.__geo_interface__],
                transform=out_transform,
                out_shape=(H, W),
                invert=False
            )
            inside_poly = ~outside_poly

            masked = out_img.mask[0]  # True invalid (outside raster)
            invalid_inside = masked & inside_poly

            if invalid_inside.any():
                skipped.append((win_id, f"incomplete_inside={int(invalid_inside.sum())}"))
                continue

            # OK — fully covered inside the polygon
            nodata = chosen_ds.nodata
            if nodata is None:
                nodata = 0
            data_to_write = out_img.filled(nodata)

            dtype_kind = np.dtype(data_to_write.dtype).kind
            predictor = 3 if dtype_kind == "f" else 2

            profile = chosen_ds.profile.copy()
            profile.update(
                height=data_to_write.shape[1],
                width=data_to_write.shape[2],
                transform=out_transform,
                driver="GTiff",
                tiled=True,
                compress="deflate",
                predictor=predictor,
                nodata=nodata,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data_to_write)
                dst.scales = chosen_ds.scales
                dst.offsets = chosen_ds.offsets
                dst.update_tags(**chosen_ds.tags())
                for b in range(1, chosen_ds.count + 1):
                    dst.update_tags(b, **chosen_ds.tags(b))

            kept.append({"window_id": win_id, "vhr_path": str(out_path)})

    finally:
        for ds in dsets:
            ds.close()

    return kept, skipped

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
        raise RuntimeError("Please set env vars CDSE_USER and CDSE_PASS.")

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
            per_ds_summary.append({"datastrip": ds, "split": split, "windows": nwin, "kept": 0, "skipped": nwin, "error": "no_product_name"})
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
                cog_name, token, 
                refresh_token_fn=lambda: get_access_token(CDSE_USERNAME, CDSE_PASSWORD)
                )
            log_line(f"UUID: {uuid_id}")

            tif_paths = download_and_extract_tifs(uuid_id, cog_name, token, ds_tmp_dir)
            print(f"Extracted {len(tif_paths)} GeoTIFF(s)")

            out_vhr_dir = OUT_ROOT / split / "vhr"

            kept, skipped = crop_vhr_windows_by_geometry_filter(
                tif_paths, ds_rows, out_vhr_dir)

            print(f"Kept windows: {len(kept)} / {nwin}")
            if skipped:
                print("Skipped:", skipped)

            if kept:
                kept_ids = {k["window_id"] for k in kept}
                kept_df = ds_rows[ds_rows["window_id"].isin(kept_ids)].copy()
                kept_meta = pd.DataFrame(kept)
                kept_df = kept_df.merge(kept_meta, on="window_id", how="left")
                kept_rows_all.append(kept_df)

            per_ds_summary.append({
                "datastrip": ds,
                "split": split,
                "windows": nwin,
                "kept": int(len(kept)),
                "skipped": int(nwin - len(kept)),
                "error": "",
            })

            log_line(f"OK datastrip={ds} kept={len(kept)} skipped={len(skipped)}")
            mark_done(ds)


        except Exception as e:
            log_line(f"[ERROR] datastrip={ds} err={e}")
            per_ds_summary.append({"datastrip": ds, "split": split, "windows": nwin, "kept": 0, "skipped": nwin, "error": str(e)})

        finally:
            shutil.rmtree(ds_tmp_dir, ignore_errors=True)

    # Save filtered parquet with only successfully-kept windows
    if kept_rows_all:
        kept_all = pd.concat(kept_rows_all, ignore_index=True)
    else:
        kept_all = df.iloc[0:0].copy()  # empty with same columns

    kept_all.to_parquet(OUT_KEPT_PARQUET, index=False)
    print(f"\nSaved kept windows parquet: {OUT_KEPT_PARQUET}  rows={len(kept_all)}")

    # Optional: save per-datastrip summary CSV
    summary_df = pd.DataFrame(per_ds_summary)
    summary_df.to_csv("vhr_download_summary.csv", index=False)
    print("Saved summary: vhr_download_summary.csv")

if __name__ == "__main__":
    main()
