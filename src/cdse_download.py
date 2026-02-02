import os
import requests
import zipfile
from pathlib import Path
from .config import DOWNLOAD_URL

def download_and_extract_tifs(prod_id: str, prod_name: str, token: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{prod_name}.zip"
    url = f"{DOWNLOAD_URL}({prod_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    #print(f"Downloading {prod_name} ...")
    with requests.get(url, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Extract only TIF/ TIFF
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.filename.lower().endswith((".tif", ".tiff")):
                target = out_dir / Path(member.filename).name
                # print(f"  Extracting {member.filename} -> {target}")
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    # try:
    #     #os.remove(zip_path)
    # except OSError:
    #     pass
