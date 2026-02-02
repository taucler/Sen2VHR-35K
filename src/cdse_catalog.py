import requests
from shapely.geometry import shape
from .config import CATALOGUE_URL, DATASET_FULL, DATE_START, DATE_END, PAGE_SIZE
from shapely.geometry.base import BaseGeometry

def build_query_url(aoi_wkt: str, skip: int = 0) -> str:
    # OData Intersects + datasetFull + time
    filter_parts = [
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'datasetFull' and "
        f"att/OData.CSC.StringAttribute/Value eq '{DATASET_FULL}'"
        ")",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
        f"ContentDate/Start ge {DATE_START}",
        f"ContentDate/Start le {DATE_END}",
    ]
    odata_filter = " and ".join(filter_parts)
    url = (
        f"{CATALOGUE_URL}?$filter={odata_filter}"
        f"&$select=Id,Name,GeoFootprint,ContentDate"
        f"&$top={PAGE_SIZE}&$skip={skip}"
        f"&$orderby=ContentDate/Start"
    )
    return url

def find_products_for_aoi(aoi_geom: BaseGeometry, token: str):
    """
    Returns:
      full_cover_product: (prod_id, prod_name) or None
      intersecting_products: list[(prod_id, prod_name)]
    """
    headers = {"Authorization": f"Bearer {token}"}
    aoi_wkt = aoi_geom.wkt

    full_cover_product = None
    intersecting_products = []
    skip = 0

    while True:
        url = build_query_url(aoi_wkt, skip=skip)
        # print(f"Querying: {url}")
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        products = data.get("value", [])
        if not products:
            break

        for prod in products:
            prod_id = prod["Id"]
            prod_name = prod["Name"]
            acq_ts = prod["ContentDate"]["Start"]
            geo = prod.get("GeoFootprint")

            if not geo:
                continue

            prod_geom = shape(geo)

            if not prod_geom.intersects(aoi_geom):
                continue

            if "_COG" not in prod_name:
                continue

            intersecting_products.append((prod_id, prod_name, acq_ts))

            if prod_geom.contains(aoi_geom) and full_cover_product is None:
                full_cover_product = (prod_id, prod_name, acq_ts)
                print(f"Found full-coverage COG candidate: {prod_name}")

        skip += len(products)

    return full_cover_product, intersecting_products
