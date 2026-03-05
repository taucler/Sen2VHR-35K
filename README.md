# Sen2VHR-35K
A Sentinel-2 to Very-High-Resolution Super-Resolution Dataset

This dataset elaboration is based on the **Optical VHR coverage over Europe (VHR_IMAGE_2024)** dataset, available under registration request via the Copernicus Data Space Ecosystem (CDSE), following this [link](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/ccm-how-to-register).

Once the registration request is approved by CDSE User Services (except for the Public user category), you can [sign in](https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=account-console&redirect_uri=https%3A%2F%2Fidentity.dataspace.copernicus.eu%2Fauth%2Frealms%2FCDSE%2Faccount%2F%23%2F&state=3d9be423-64ad-422c-8721-a16ba2d4bfe4&response_mode=fragment&response_type=code&scope=openid&nonce=1fe7ace4-3400-4cd8-b396-25e2a2254e05&code_challenge=6-di5BpgOLzlI-0PwbX8adCkS9D3nBXiiD-YyF0Ay6o&code_challenge_method=S256) to Copernicus Data Space Ecosystem and download data. 

Before downloading any VHR imagery, this workflow relies exclusively on freely accessible reference data to define and structure the dataset. In particular, we use the official reference shapefiles provided by CDSE, which include the spatial footprints and metadata of the VHR_IMAGE_2024 products, freely available [here](https://s3.waw3-1.cloudferro.com/swift/v1/portal_uploads_prod/VHR2024-REF-PACKAGE-20241122.zip).

Using these reference layers allows us to construct and validate the dataset geometry, windowing scheme, and acquisition mapping prior to any restricted data download, ensuring transparency, reproducibility, and compliance with data access constraints.

Large derived datasets (GeoPackage, Parquet) are not versioned.
They can be reproduced by running the provided notebooks and scripts, and will be made available on Hugging Face.
