#!/usr/bin/env python3
"""
Collect high-quality training data from USGS cameras.

This script:
1. Validates each camera site has reasonable gage height ranges (0-50 ft)
2. Ensures sufficient elevation variance for training
3. Filters out bad/missing data
4. Targets ~5000 images across multiple sites
5. Creates proper train/val/test splits BY SITE (not by image)

Usage:
    python collect_quality_data.py

Output:
    data/quality_dataset.csv - Combined dataset with all validated samples
    data/train_quality.csv   - Training set (sites held out)
    data/val_quality.csv     - Validation set
    data/test_quality.csv    - Test set (completely unseen sites)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Add pynims to path
sys.path.insert(0, 'packages/pynims')

try:
    from pynims.client import NIMSClient
    from pynims.utils import convert_nims_image_name_to_utc_date
except ImportError:
    print("ERROR: pynims not found. Make sure packages/pynims exists.")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target number of images
TARGET_IMAGES = 5000
IMAGES_PER_SITE = 250  # Aim for this many per site

# Quality filters
MIN_GAGE_HEIGHT = 0.0    # Minimum valid gage height (ft)
MAX_GAGE_HEIGHT = 50.0   # Maximum valid gage height (ft) - filters out elevation-based readings
MIN_ELEVATION_RANGE = 0.5  # Site must have at least this much variance in water level
MIN_IMAGES_PER_SITE = 50   # Need at least this many images to include site
TIME_TOLERANCE_MINUTES = 5  # Max time diff between image and gauge reading

# Verified camera-to-gauge mappings
# Only include sites with known good data quality
CAMERA_GAUGE_MAPPINGS = {
    # =========================================================================
    # TIER 1: Well-tested sites with good data (from original 3-site test)
    # =========================================================================
    'WI_Yahara_River_at_McFarland_GATES': '05430500',
    'MN_Mississippi_River_Abv_37th_Ave_NE_in_Fridley': '05288500',
    'PA_Allegheny_River_at_Franklin': '03025500',
    
    # =========================================================================
    # TIER 2: Additional Wisconsin sites (similar region, likely good quality)
    # =========================================================================
    'WI_Menomonee_River_at_16th_Street_at_Milwaukee': '04087142',
    'WI_Manitowoc_River_at_Manitowoc': '04085427',
    'WI_Kinnickinnic_River_near_River_Falls_SECCHI': '05342000',
    'WI_Chippewa_River_near_Bruce': '05356000',
    'WI_Red_Cedar_River_near_Colfax': '05367500',
    'WI_Bark_River_near_Rome': '05426250',
    
    # =========================================================================
    # TIER 3: Other Midwest sites
    # =========================================================================
    'MN_RUM_RIVER_NEAR_ST_FRANCIS_MN': '05286000',
    'MN_High_Island_Creek_near_Henderson': '05327000',
    'OH_Maumee_River_near_Defiance': '04192500',
    'OH_St_Marys_River_at_Walcot_Street_at_Willshire': '04177720',
    
    # =========================================================================
    # TIER 4: East Coast sites
    # =========================================================================
    'MA_Connecticut_River_near_Northfield': '01154500',
    'MD_Deer_Creek_at_Eden_Mill_Dam_near_Pylesville': '01579905',
    'NJ_Middle_Brook_at_Burnt_Mills': '01399100',
    'NJ_Stony_Brook_at_Princeton': '01401000',
    'NJ_Great_Egg_Harbor_River_at_Folsom': '01411000',
    'NY_West_Branch_Croton_River_near_Croton_Falls': '01374701',
    'PA_Swatara_Creek_near_Palmyra': '01573208',
    'VA_OPEQUON_CREEK_NEAR_BERRYVILLE': '01615000',
    'DC_Watts_Branch_at_Washington': '01651800',
    'VA_FLATLICK_BRANCH_ABOVE_FROG_BRANCH_AT_CHANTILLY': '01656903',
    
    # =========================================================================
    # TIER 5: South/Southeast sites
    # =========================================================================
    'SC_Congaree_River_below_Cayce_DOWNSTREAM_CAMERA': '021695075',
    'SC_Lake_Moultrie_Tailrace_Canal_at_Moncks_Corner': '02172002',
    'LA_Buxton_Creek_at_Hwy_27_near_DeQuincy': '08016910',
    'TX_Elm_Fk_Trinity_Rv_nr_Lewisville': '08053000',
    'TX_New_Year_Ck_at_FM_1155_nr_Chappel_Hill': '08111110',
    
    # =========================================================================
    # TIER 6: West Coast / Mountain sites (CAREFUL - check elevation ranges!)
    # =========================================================================
    'CA_Merced_River_at_Happy_Isles_Bridge_Yosemite': '11264500',
    'CA_Threemile_Slough_Nr_Rio_Vista_CA': '11337080',
    'CA_Middle_River_at_Middle_River': '11312676',
    'OR_Klamath_River_below_John_C_Boyle_Powerplant_near_Keno': '11510700',
    'WA_Pend_Oreille_River_at_Newport': '12395500',
    'ID_Lemhi_River_below_L5_Diversion_near_Salmon': '13305310',
    
    # =========================================================================
    # TIER 7: Plains / Central sites
    # =========================================================================
    'NE_Platte_River_near_Grand_Island': '06770500',
    'KY_OHIO_R_US_OF_MCALPINE_DAM_AT_RRB_AT_LOUISVILLE': '03293551',
    'LA_Black_River_at_Jonesville': '07373267',
}


def validate_site_data(df, camera_id):
    """
    Validate that a site's data is suitable for training.
    
    Returns:
        (is_valid, reason, stats)
    """
    if df is None or len(df) == 0:
        return False, "No data", {}
    
    # Check for gage height column
    gage_col = None
    for col in df.columns:
        if '00065' in col and '_cd' not in col:
            gage_col = col
            break
    
    if gage_col is None:
        return False, "No gage height column", {}
    
    # Get valid readings (not NaN, not inf)
    valid_mask = df[gage_col].notna() & ~np.isinf(df[gage_col])
    valid_readings = df.loc[valid_mask, gage_col]
    
    if len(valid_readings) == 0:
        return False, "No valid gage readings", {}
    
    # Compute stats
    min_val = valid_readings.min()
    max_val = valid_readings.max()
    mean_val = valid_readings.mean()
    std_val = valid_readings.std()
    elevation_range = max_val - min_val
    
    stats = {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'range': elevation_range,
        'n_valid': len(valid_readings),
        'gage_col': gage_col,
    }
    
    # Check bounds
    if min_val < MIN_GAGE_HEIGHT:
        return False, f"Min gage height {min_val:.1f} < {MIN_GAGE_HEIGHT}", stats
    
    if max_val > MAX_GAGE_HEIGHT:
        return False, f"Max gage height {max_val:.1f} > {MAX_GAGE_HEIGHT} (likely elevation-based)", stats
    
    if elevation_range < MIN_ELEVATION_RANGE:
        return False, f"Elevation range {elevation_range:.2f} < {MIN_ELEVATION_RANGE} (not enough variance)", stats
    
    if len(valid_readings) < MIN_IMAGES_PER_SITE:
        return False, f"Only {len(valid_readings)} samples < {MIN_IMAGES_PER_SITE} minimum", stats
    
    return True, "OK", stats


def collect_site_data(cam_id, site_id, max_images=IMAGES_PER_SITE):
    """
    Collect and validate data for a single camera site.
    
    Returns:
        DataFrame with validated samples, or None if site is invalid
    """
    print(f'\n{"="*70}')
    print(f'Camera: {cam_id}')
    print(f'USGS Site: {site_id}')
    print(f'{"="*70}')
    
    try:
        client = NIMSClient()
    except Exception as e:
        print(f'   Failed to create NIMS client: {e}')
        return None
    
    # Create directories
    base_dir = Path(f'data/{cam_id}')
    images_dir = base_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Get image list
    print(f'\n1. Fetching image list...')
    try:
        images = client.get_image_list(cam_id, recursive=True, max_results=max_images * 2)
    except Exception as e:
        print(f'   Error getting images: {e}')
        client.close()
        return None
    
    print(f'   Found {len(images)} images available')
    
    if len(images) < MIN_IMAGES_PER_SITE:
        print(f'   Not enough images (need {MIN_IMAGES_PER_SITE})')
        client.close()
        return None
    
    # Step 2: Download images
    print(f'\n2. Downloading up to {max_images} images...')
    downloaded = []
    for i, img_name in enumerate(images[:max_images]):
        try:
            local_path = images_dir / img_name
            if not local_path.exists():
                client.download_image(img_name, str(images_dir))
            downloaded.append(img_name)
            
            if (i + 1) % 50 == 0:
                print(f'   Downloaded {i + 1}/{min(len(images), max_images)}...')
        except Exception as e:
            pass
    
    print(f'   ✓ Downloaded {len(downloaded)} images')
    client.close()
    
    if len(downloaded) < MIN_IMAGES_PER_SITE:
        print(f'   Not enough downloaded (need {MIN_IMAGES_PER_SITE})')
        return None
    
    # Step 3: Parse timestamps
    print(f'\n3. Parsing timestamps...')
    image_data = []
    for img_name in downloaded:
        try:
            timestamp = convert_nims_image_name_to_utc_date(img_name)
            image_data.append({
                'image_name': img_name,
                'timestamp': timestamp,
            })
        except Exception as e:
            pass
    
    print(f'   ✓ Parsed {len(image_data)} timestamps')
    
    if len(image_data) < MIN_IMAGES_PER_SITE:
        print(f'   Not enough valid timestamps')
        return None
    
    # Get date range
    timestamps = [d['timestamp'] for d in image_data]
    min_date = min(timestamps)
    max_date = max(timestamps)
    print(f'   Date range: {min_date.date()} to {max_date.date()}')
    
    # Step 4: Fetch gauge data
    print(f'\n4. Fetching gauge data from USGS...')
    try:
        import dataretrieval.nwis as nwis
        import pandas as pd
        
        start_date = (min_date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        df, meta = nwis.get_iv(
            sites=site_id,
            parameterCd=['00065', '00060'],  # gage height and discharge
            start=start_date,
            end=end_date,
        )
        
        if df is None or len(df) == 0:
            print(f'   No gauge data found!')
            return None
        
        df = df.reset_index()
        print(f'   ✓ Found {len(df)} gauge readings')
        
    except Exception as e:
        print(f'   Error fetching gauge data: {e}')
        return None
    
    # Step 5: Validate site data quality
    print(f'\n5. Validating data quality...')
    is_valid, reason, stats = validate_site_data(df, cam_id)
    
    if not is_valid:
        print(f'   Site rejected: {reason}')
        if stats:
            print(f'      Stats: min={stats.get("min", "N/A"):.2f}, max={stats.get("max", "N/A"):.2f}, range={stats.get("range", "N/A"):.2f}')
        return None
    
    print(f'   ✓ Site validated!')
    print(f'      Gage height: {stats["min"]:.2f} - {stats["max"]:.2f} ft (range: {stats["range"]:.2f} ft)')
    print(f'      Mean: {stats["mean"]:.2f} ft, Std: {stats["std"]:.2f} ft')
    
    gage_col = stats['gage_col']
    
    # Step 6: Merge image and gauge data
    print(f'\n6. Merging image and gauge data...')
    import pandas as pd
    
    img_df = pd.DataFrame(image_data)
    img_df['timestamp'] = pd.to_datetime(img_df['timestamp'], utc=True)
    
    # Find datetime column in gauge data
    datetime_col = [c for c in df.columns if 'datetime' in c.lower()][0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.rename(columns={datetime_col: 'gauge_time'})
    
    # Sort both
    img_df = img_df.sort_values('timestamp')
    df = df.sort_values('gauge_time')
    
    # Merge using nearest timestamp
    merged = pd.merge_asof(
        img_df,
        df,
        left_on='timestamp',
        right_on='gauge_time',
        direction='nearest',
        tolerance=pd.Timedelta(f'{TIME_TOLERANCE_MINUTES}min')
    )
    
    # Drop rows without matched gauge data
    merged = merged.dropna(subset=['gauge_time'])
    
    # Add metadata
    merged['camera_id'] = cam_id
    merged['site_id'] = site_id
    merged['image_path'] = merged['image_name'].apply(lambda x: f'data/{cam_id}/images/{x}')
    
    # Rename gage height column for consistency
    merged = merged.rename(columns={gage_col: 'gage_height_ft'})
    
    # Filter to valid gage heights only
    merged = merged[
        (merged['gage_height_ft'] >= MIN_GAGE_HEIGHT) & 
        (merged['gage_height_ft'] <= MAX_GAGE_HEIGHT) &
        (merged['gage_height_ft'].notna())
    ]
    
    print(f'   ✓ Matched {len(merged)} images with gauge data')
    
    if len(merged) < MIN_IMAGES_PER_SITE:
        print(f'   Not enough matched samples (need {MIN_IMAGES_PER_SITE})')
        return None
    
    # Save site data
    output_file = base_dir / 'images_and_data.csv'
    merged.to_csv(output_file, index=False)
    print(f'   ✓ Saved to {output_file}')
    
    return merged


def create_splits(combined_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits BY SITE (not by image).
    
    This ensures the model is tested on completely unseen cameras.
    """
    print('\n' + '='*70)
    print('Creating train/val/test splits BY SITE')
    print('='*70)
    
    sites = combined_df['camera_id'].unique()
    n_sites = len(sites)
    
    # Shuffle sites
    np.random.seed(42)
    np.random.shuffle(sites)
    
    # Split sites
    n_train = int(n_sites * train_ratio)
    n_val = int(n_sites * val_ratio)
    
    train_sites = sites[:n_train]
    val_sites = sites[n_train:n_train + n_val]
    test_sites = sites[n_train + n_val:]
    
    print(f'\nSite allocation:')
    print(f'  Train sites ({len(train_sites)}): {list(train_sites)}')
    print(f'  Val sites ({len(val_sites)}): {list(val_sites)}')
    print(f'  Test sites ({len(test_sites)}): {list(test_sites)}')
    
    # Create splits
    train_df = combined_df[combined_df['camera_id'].isin(train_sites)].copy()
    val_df = combined_df[combined_df['camera_id'].isin(val_sites)].copy()
    test_df = combined_df[combined_df['camera_id'].isin(test_sites)].copy()
    
    print(f'\nSample counts:')
    print(f'  Train: {len(train_df)} images from {len(train_sites)} sites')
    print(f'  Val: {len(val_df)} images from {len(val_sites)} sites')
    print(f'  Test: {len(test_df)} images from {len(test_sites)} sites')
    
    return train_df, val_df, test_df


def main():
    print('='*70)
    print('🌊 QUALITY DATA COLLECTION FOR WATER LEVEL ESTIMATION')
    print('='*70)
    print(f'\nTarget: ~{TARGET_IMAGES} images from {len(CAMERA_GAUGE_MAPPINGS)} potential sites')
    print(f'Quality filters:')
    print(f'  - Gage height range: {MIN_GAGE_HEIGHT} - {MAX_GAGE_HEIGHT} ft')
    print(f'  - Min elevation variance: {MIN_ELEVATION_RANGE} ft')
    print(f'  - Min images per site: {MIN_IMAGES_PER_SITE}')
    print(f'  - Time tolerance: {TIME_TOLERANCE_MINUTES} minutes')
    
    all_data = []
    successful_sites = []
    failed_sites = []
    site_stats = {}
    
    for cam_id, site_id in CAMERA_GAUGE_MAPPINGS.items():
        try:
            df = collect_site_data(cam_id, site_id, max_images=IMAGES_PER_SITE)
            
            if df is not None and len(df) > 0:
                all_data.append(df)
                successful_sites.append(cam_id)
                
                # Track stats
                site_stats[cam_id] = {
                    'samples': len(df),
                    'min_elev': df['gage_height_ft'].min(),
                    'max_elev': df['gage_height_ft'].max(),
                    'mean_elev': df['gage_height_ft'].mean(),
                    'std_elev': df['gage_height_ft'].std(),
                }
                
                print(f'\n {cam_id}: {len(df)} samples')
                
                # Check if we have enough
                total_so_far = sum(len(d) for d in all_data)
                if total_so_far >= TARGET_IMAGES:
                    print(f'\n🎯 Reached target of {TARGET_IMAGES} images!')
                    break
            else:
                failed_sites.append(cam_id)
                print(f'\n {cam_id}: Rejected')
                
        except Exception as e:
            failed_sites.append(cam_id)
            print(f'\n {cam_id}: Error - {e}')
    
    # Combine all data
    if not all_data:
        print('\n No data collected!')
        return
    
    import pandas as pd
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save combined dataset
    combined.to_csv('data/quality_dataset.csv', index=False)
    
    # Create splits
    train_df, val_df, test_df = create_splits(combined)
    
    train_df.to_csv('data/train_quality.csv', index=False)
    val_df.to_csv('data/val_quality.csv', index=False)
    test_df.to_csv('data/test_quality.csv', index=False)
    
    # Print summary
    print('\n' + '='*70)
    print(' COLLECTION SUMMARY')
    print('='*70)
    print(f'\nSuccessful sites: {len(successful_sites)}')
    print(f'Failed sites: {len(failed_sites)}')
    print(f'Total samples: {len(combined)}')
    
    print(f'\nElevation range across all sites:')
    print(f'  Min: {combined["gage_height_ft"].min():.2f} ft')
    print(f'  Max: {combined["gage_height_ft"].max():.2f} ft')
    print(f'  Mean: {combined["gage_height_ft"].mean():.2f} ft')
    print(f'  Std: {combined["gage_height_ft"].std():.2f} ft')
    
    print(f'\nSamples per site:')
    for site, stats in sorted(site_stats.items(), key=lambda x: -x[1]['samples']):
        print(f'  {site}: {stats["samples"]} samples '
              f'(elev: {stats["min_elev"]:.1f}-{stats["max_elev"]:.1f} ft)')
    
    print(f'\nFiles saved:')
    print(f'  data/quality_dataset.csv - Combined dataset ({len(combined)} samples)')
    print(f'  data/train_quality.csv   - Training set ({len(train_df)} samples)')
    print(f'  data/val_quality.csv     - Validation set ({len(val_df)} samples)')
    print(f'  data/test_quality.csv    - Test set ({len(test_df)} samples)')
    
    # Save metadata
    metadata = {
        'collection_date': datetime.now().isoformat(),
        'total_samples': len(combined),
        'successful_sites': successful_sites,
        'failed_sites': failed_sites,
        'site_stats': site_stats,
        'config': {
            'min_gage_height': MIN_GAGE_HEIGHT,
            'max_gage_height': MAX_GAGE_HEIGHT,
            'min_elevation_range': MIN_ELEVATION_RANGE,
            'min_images_per_site': MIN_IMAGES_PER_SITE,
            'time_tolerance_minutes': TIME_TOLERANCE_MINUTES,
        },
        'splits': {
            'train_sites': list(train_df['camera_id'].unique()),
            'val_sites': list(val_df['camera_id'].unique()),
            'test_sites': list(test_df['camera_id'].unique()),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
        }
    }
    
    with open('data/collection_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f'  data/collection_metadata.json - Collection metadata')
    
    print('\n' + '='*70)
    print(' COLLECTION COMPLETE!')
    print('='*70)


if __name__ == '__main__':
    main()