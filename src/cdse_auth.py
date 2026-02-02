import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_access_token(username=None, password=None) -> str:
    username = username or os.getenv("CDSE_USERNAME")
    password = password or os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError("CDSE_USERNAME / CDSE_PASSWORD not set.")

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(os.getenv("TOKEN_URL"), data=data)
    r.raise_for_status()
    return r.json()["access_token"]
