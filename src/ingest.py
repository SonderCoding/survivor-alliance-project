import pandas as pd
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/stiles/survivoR2py/main/data/processed/csv"
TABLES = ["vote_history", "castaways", "boot_mapping", "season_summary"]
RAW_DIR = Path("data/raw")

def fetch_table(name: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{name}.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.csv"
    out_path.write_bytes(resp.content)
    df = pd.read_csv(out_path)
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} cols -> {out_path}")
    return df

if __name__ == "__main__":
    for table in TABLES:
        fetch_table(table)