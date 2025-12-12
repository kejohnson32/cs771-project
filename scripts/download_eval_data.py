#!/usr/bin/env python3
"""
Download evaluation data for SEEN vs UNSEEN.

SEEN  = NEW images from the 14 training sites (same cameras, different dates)
UNSEEN = data/test_quality.csv (4 unseen cameras, images already on disk)

This script:

1. Reads data/train_quality.csv to get (camera_id, site_no) for training sites
2. For each site:
      - pulls recent gage height time-series from NWIS (parameter 00065)
      - pulls recent image metadata from NIMS
      - matches each image timestamp to the closest gage height
      - downloads the image to data/seen_eval_images/<camera_id>/
      - writes one row per image to data/seen_eval.csv
3. Copies data/test_quality.csv to data/unseen_eval.csv for UNSEEN eval.

Run from anywhere:

    cd cs771-project
    source .venv/bin/activate
    python scripts/download_eval_data.py
"""

import os
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Project root = parent of this scripts/ directory
ROOT = Path(__file__).resolve().parents[1]

# USGS endpoints
NIMS_API = "https://api.waterdata.usgs.gov/nims/v0"
WATERSERVICES_API = "https://waterservices.usgs.gov/nwis/iv/"

# Paths
DATA_DIR = ROOT / "data"
SEEN_IMAGES_DIR = DATA_DIR / "seen_eval_images"
SEEN_CSV = DATA_DIR / "seen_eval.csv"
UNSEEN_CSV = DATA_DIR / "unseen_eval.csv"
TRAIN_CSV = DATA_DIR / "train_quality.csv"
TEST_CSV = DATA_DIR / "test_quality.csv"

# How many NEW images per training site
IMAGES_PER_SITE = 30


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_site_info_from_train():
    """
    Return mapping: camera_id -> site_no (str, padded to 8 digits)
    using data/train_quality.csv.
    """
    train_df = pd.read_csv(TRAIN_CSV)

    site_info = {}
    for camera_id in train_df["camera_id"].unique():
        if camera_id == "camera_id":  # guard in case of weird header row
            continue
        site_df = train_df[train_df["camera_id"] == camera_id]
        site_no = str(int(site_df["site_no"].iloc[0])).zfill(8)
        site_info[camera_id] = site_no

    return site_info


def fetch_gage_height(site_no: str, start_date: str, end_date: str):
    """
    Fetch gage height readings (parameterCd=00065) for [start_date, end_date].

    Returns dict: {timestamp_str -> gage_height_ft}
    """
    params = {
        "sites": site_no,
        "parameterCd": "00065",
        "startDT": start_date,
        "endDT": end_date,
        "format": "json",
    }

    try:
        resp = requests.get(WATERSERVICES_API, params=params, timeout=60)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        series = data.get("value", {}).get("timeSeries", [])
        if not series:
            return {}

        readings = series[0].get("values", [{}])[0].get("value", [])
        out = {}
        for r in readings:
            v = r.get("value")
            if v is None:
                continue
            val = float(v)
            if val <= 0 or val > 50:
                continue
            out[r["dateTime"]] = val
        return out
    except Exception as e:
        print(f"    Error fetching gage data for {site_no}: {e}")
        return {}


def fetch_image_list(site_no: str, start_date: str, end_date: str, limit: int = 100):
    """
    Fetch image metadata from NIMS for given site and date range.

    Returns a list of JSON objects (one per image) or [] on failure.
    """
    params = {
        "siteCodes": site_no,
        "startDT": start_date,
        "endDT": end_date,
        "limit": limit,
    }

    try:
        resp = requests.get(f"{NIMS_API}/images", params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"    Error fetching image list for {site_no}: {e}")
        return []


def download_image(url: str, filepath: Path) -> bool:
    """Download image URL to filepath. Returns True on success."""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"      Error downloading {url}: {e}")
    return False


def match_gage_to_image(img_time: str, gage_data: dict, max_diff_minutes: int = 15):
    """
    Find the closest gage_height within ±max_diff_minutes of image timestamp.
    Returns gage_height_ft or None.
    """
    img_dt = pd.to_datetime(img_time)
    best_val = None
    min_diff = timedelta(minutes=max_diff_minutes)

    for gage_dt, gage_val in gage_data.items():
        diff = abs(img_dt - pd.to_datetime(gage_dt))
        if diff < min_diff:
            min_diff = diff
            best_val = gage_val

    return best_val


def download_site_images(
    camera_id: str,
    site_no: str,
    output_dir: Path,
    existing_paths: set[str],
    max_images: int = 30,
):
    """
    Download NEW images for a single training site.

    - looks at last 90 days
    - skips images we've already seen in train_quality (by URL)
    - only keeps images that can be matched to a positive gage_height_ft
    """
    print(f"\n  {camera_id} (site {site_no})")

    site_dir = output_dir / camera_id
    site_dir.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # 1) gage data
    print("    Fetching gage data...")
    gage_data = fetch_gage_height(
        site_no, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
    if not gage_data:
        print("    No gage data found, skipping site.")
        return []

    print(f"    Found {len(gage_data)} gage readings")

    # 2) image list
    print("    Fetching image list...")
    images = fetch_image_list(
        site_no,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        limit=max_images * 3,
    )
    if not images:
        print("    No images found for this site.")
        return []

    print(f"    Found {len(images)} images in NIMS metadata")

    samples = []
    downloaded = 0

    for img_info in images:
        if downloaded >= max_images:
            break

        img_url = img_info.get("url") or img_info.get("imageUrl")
        img_dt = img_info.get("dateTime") or img_info.get("timestamp")
        if not img_url or not img_dt:
            continue

        # avoid duplicates already in dataset
        if any(img_url in p for p in existing_paths):
            continue

        # match gage height
        gage_val = match_gage_to_image(img_dt, gage_data)
        if gage_val is None or gage_val <= 0 or gage_val > 50:
            continue

        img_time = pd.to_datetime(img_dt)
        filename = f"{camera_id}___{img_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.jpg"
        filepath = site_dir / filename

        if filepath.exists():
            # already downloaded before
            downloaded += 1
            samples.append(
                {
                    "camera_id": camera_id,
                    "site_no": site_no,
                    "image_path": str(filepath),
                    "datetime": str(img_time),
                    "gage_height_ft": gage_val,
                    "image_name": filename,
                }
            )
            continue

        if download_image(img_url, filepath):
            downloaded += 1
            samples.append(
                {
                    "camera_id": camera_id,
                    "site_no": site_no,
                    "image_path": str(filepath),
                    "datetime": str(img_time),
                    "gage_height_ft": gage_val,
                    "image_name": filename,
                }
            )

        time.sleep(0.1)  # rate-limit slightly

    print(f"    Downloaded {downloaded} new images")
    return samples


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DOWNLOAD EVALUATION DATA FOR SEEN VS UNSEEN")
    print("=" * 70)

    DATA_DIR.mkdir(exist_ok=True)
    SEEN_IMAGES_DIR.mkdir(exist_ok=True)

    # --- sanity checks ---
    if not TRAIN_CSV.exists():
        print(f"\nERROR: {TRAIN_CSV} not found!")
        print("Please make sure data/train_quality.csv exists.")
        return

    if not TEST_CSV.exists():
        print(f"\nWARNING: {TEST_CSV} not found.")
        print("UNSEEN evaluation CSV will NOT be created.")
    else:
        print(f"Found {TEST_CSV} (for UNSEEN sites).")

    # --- training sites (SEEN) ---
    site_info = get_site_info_from_train()
    print(f"\nFound {len(site_info)} training sites (SEEN sites):")
    for cam, site in site_info.items():
        print(f"  - {cam} ({site})")

    # existing paths, so we don't re-download same URLs
    train_df = pd.read_csv(TRAIN_CSV)
    existing_paths = set(train_df["image_path"].dropna().tolist())
    print(f"\nExisting training images in CSV: {len(existing_paths)}")

    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOADING NEW IMAGES FOR SEEN SITES")
    print("=" * 70)

    all_samples = []
    for camera_id, site_no in tqdm(site_info.items(), desc="Sites"):
        samples = download_site_images(
            camera_id,
            site_no,
            SEEN_IMAGES_DIR,
            existing_paths,
            max_images=IMAGES_PER_SITE,
        )
        all_samples.extend(samples)
        time.sleep(0.5)  # rate-limit per site

    if all_samples:
        seen_df = pd.DataFrame(all_samples)
        seen_df.to_csv(SEEN_CSV, index=False)
        print(f"\n✓ Created {SEEN_CSV}")
        print(
            f"  {len(seen_df)} images from "
            f"{seen_df['camera_id'].nunique()} training sites"
        )
    else:
        print("\n⚠ No new images downloaded for SEEN eval.")
        print("  Falling back to val_quality.csv if available.")
        val_csv = DATA_DIR / "val_quality.csv"
        if val_csv.exists():
            shutil.copy(val_csv, SEEN_CSV)
            print(f"  Copied {val_csv} -> {SEEN_CSV}")

    # --- UNSEEN eval: just copy test_quality.csv ---
    print("\n" + "=" * 70)
    print("STEP 2: PREPARE UNSEEN EVAL CSV")
    print("=" * 70)

    if TEST_CSV.exists():
        shutil.copy(TEST_CSV, UNSEEN_CSV)
        unseen_df = pd.read_csv(UNSEEN_CSV)
        print(f"\n✓ Created {UNSEEN_CSV}")
        print(
            f"  {len(unseen_df)} images from "
            f"{unseen_df['camera_id'].nunique()} unseen sites"
        )
    else:
        print("  Skipping UNSEEN eval CSV creation (no test_quality.csv).")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Output files in {DATA_DIR}/:

  SEEN evaluation:
    - {SEEN_CSV.name}
    - {SEEN_IMAGES_DIR.name}/  (downloaded images)

  UNSEEN evaluation:
    - {UNSEEN_CSV.name}   (copy of test_quality.csv)

Next:
  1. Run:  python scripts/eval_seen_unseen.py
  2. That script will load these CSVs and evaluate each model,
     producing scatter plots and a JSON summary in outputs/.
""")


if __name__ == "__main__":
    main()
