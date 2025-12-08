#!/usr/bin/env python3
"""
Collect training data from USGS cameras with known gauge station mappings.
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, 'packages/pynims')
from pynims.client import NIMSClient
from pynims.utils import convert_nims_image_name_to_utc_date

# Known camera ID to USGS site ID mappings
# These are cameras where we know the corresponding gauge station
CAMERA_GAUGE_MAPPINGS = {
    # Wisconsin
    'WI_Yahara_River_at_McFarland_GATES': '05430500',
    'WI_Menomonee_River_at_16th_Street_at_Milwaukee': '04087142',
    'WI_Manitowoc_River_at_Manitowoc': '04085427',
    'WI_Kinnickinnic_River_near_River_Falls_SECCHI': '05342000',
    'WI_Chippewa_River_near_Bruce': '05356000',
    'WI_Red_Cedar_River_near_Colfax': '05367500',
    
    # Minnesota
    'MN_Mississippi_River_Abv_37th_Ave_NE_in_Fridley': '05288500',
    'MN_RUM_RIVER_NEAR_ST_FRANCIS_MN': '05286000',
    'MN_High_Island_Creek_near_Henderson': '05327000',
    
    # Other states
    'PA_Allegheny_River_at_Franklin': '03025500',
    'CA_Merced_River_at_Happy_Isles_Bridge_Yosemite': '11264500',
    'NE_Platte_River_near_Grand_Island': '06770500',
    'MA_Connecticut_River_near_Northfield': '01154500',

    #7 New Cameras (Adding For Test 1 (12/6))
    'MD_Deer_Creek_at_Eden_Mill_Dam_near_Pylesville' : '01579905',
    'OR_Klamath_River_below_John_C_Boyle_Powerplant_near_Keno': '11510700',
    'WI_Green_Bay_Oil_Depot': '040851385',
    'NY_Cannonsville_Reservoir_Diversion_Channel_near_Grahamsville': '01365100',
    'NJ_Middle_Brook_at_Burnt_Mills': '01399100',
    'WI_Bark_River_near_Rome': '05426250',
    'AK_Bradly_River_near_Tidewater_near_Homer': '15239070',

    #20 new cameras 
    'VA_OPEQUON_CREEK_NEAR_BERRYVILLE': '01615000',
    'CO_White_River_below_Boise_Creek_near_Rangely': '09306290',
    'DC_Watts_Branch_at_Washington': '01651800',
    'WA_Pend_Oreille_River_at_Newport': '12395500',
    'ID_Lemhi_River_below_L5_Diversion_near_Salmon': '13305310',
    'PA_Swatara_Creek_near_Palmyra': '01573208',
    'TX_Elm_Fk_Trinity_Rv_nr_Lewisville': '08053000',
    'NY_West_Branch_Croton_River_near_Croton_Falls': '01374701',
    'LA_Buxton_Creek_at_Hwy_27_near_DeQuincy': '08016910',
    'TX_New_Year_Ck_at_FM_1155_nr_Chappel_Hill': '08111110',
    'SC_Congaree_River_below_Cayce_DOWNSTREAM_CAMERA': '021695075',
    'CA_Threemile_Slough_Nr_Rio_Vista_CA': '11337080',
    'LA_Black_River_at_Jonesville': '07373267',
    'SC_Lake_Moultrie_Tailrace_Canal_at_Moncks_Corner': '02172002',
    'NJ_Stony_Brook_at_Princeton': '01401000',
    'PA_Swatara_Creek_near_Palmyra': '01573208',
    'NJ_Great_Egg_Harbor_River_at_Folsom': '01411000',
    'CA_Middle_River_at_Middle_River': '11312676',
    'ID_Big_Wood_River_at_Hailey_Total_Flow': '13139510',
    'KY_OHIO_R_US_OF_MCALPINE_DAM_AT_RRB_AT_LOUISVILLE': '03293551',
    'VA_FLATLICK_BRANCH_ABOVE_FROG_BRANCH_AT_CHANTILLY': '01656903',
    'OH_Maumee_River_near_Defiance': '04192500',
    'UT_Weber_River_at_Gateway_UTAH': '10136500'


}


def collect_data_for_camera(cam_id, site_id, num_images=200):
    """Collect images and gauge data for a single camera."""
    
    print(f'\n{"="*60}')
    print(f'Collecting data for: {cam_id}')
    print(f'USGS Site: {site_id}')
    print(f'{"="*60}')
    
    client = NIMSClient()
    
    # Create directories
    base_dir = f'data/{cam_id}'
    images_dir = f'{base_dir}/images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Step 1: Get image list
    print(f'\n1. Fetching image list...')
    images = client.get_image_list(cam_id, recursive=True, max_results=num_images)
    print(f'   Found {len(images)} images')
    
    if len(images) < 50:
        print(f'   Not enough images, skipping')
        client.close()
        return None
    
    # Step 2: Download images
    print(f'\n2. Downloading {min(len(images), num_images)} images...')
    downloaded = []
    for i, img_name in enumerate(images[:num_images]):
        try:
            client.download_image(img_name, images_dir)
            downloaded.append(img_name)
            if (i + 1) % 25 == 0:
                print(f'   Downloaded {i + 1}/{min(len(images), num_images)}...')
        except Exception as e:
            pass
    
    print(f'   Downloaded {len(downloaded)} images')
    client.close()
    
    # Step 3: Parse timestamps from image names
    print(f'\n3. Parsing image timestamps...')
    image_data = []
    for img_name in downloaded:
        try:
            timestamp = convert_nims_image_name_to_utc_date(img_name)
            image_data.append({
                'image_name': img_name,
                'timestamp': timestamp.isoformat(),
            })
        except Exception as e:
            pass
    
    print(f'   Parsed {len(image_data)} timestamps')
    
    if not image_data:
        print('   No valid timestamps, skipping')
        return None
    
    # Get date range
    timestamps = [datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) for d in image_data]
    min_date = min(timestamps)
    max_date = max(timestamps)
    print(f'   Date range: {min_date.date()} to {max_date.date()}')
    
    # Step 4: Fetch gauge data
    print(f'\n4. Fetching gauge data from USGS...')
    try:
        import dataretrieval.nwis as nwis
        import pandas as pd
        
        # Add buffer to date range
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
        print(f'   Found {len(df)} gauge readings')
        print(f'   Columns: {list(df.columns)}')
        
        # Save gauge data
        gauge_file = f'{base_dir}/gauge_data.csv'
        df.to_csv(gauge_file, index=False)
        print(f'   Saved to {gauge_file}')
        
    except Exception as e:
        print(f'   Error fetching gauge data: {e}')
        return None
    
    # Step 5: Merge image and gauge data
    print(f'\n5. Merging image and gauge data...')
    
    import pandas as pd
    
    # Create image dataframe
    img_df = pd.DataFrame(image_data)
    img_df['timestamp'] = pd.to_datetime(img_df['timestamp'], utc=True)
    
    # Find datetime column in gauge data
    datetime_col = [c for c in df.columns if 'datetime' in c.lower()][0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.rename(columns={datetime_col: 'gauge_time'})
    
    # Sort both
    img_df = img_df.sort_values('timestamp')
    df = df.sort_values('gauge_time')
    
    # Merge using nearest timestamp (within 5 minutes)
    merged = pd.merge_asof(
        img_df,
        df,
        left_on='timestamp',
        right_on='gauge_time',
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )
    
    # Drop rows without gauge data
    merged = merged.dropna(subset=['gauge_time'])
    
    # Add image path
    merged['image_path'] = merged['image_name'].apply(lambda x: f'images/{x}')
    
    # Add camera/site info
    merged['camera_id'] = cam_id
    merged['site_id'] = site_id
    
    print(f'   Matched {len(merged)} images with gauge data')
    
    if len(merged) < 20:
        print('   Not enough matched data, skipping')
        return None
    
    # Find elevation column
    elev_col = None
    for col in merged.columns:
        if '00065' in col and '_cd' not in col:
            elev_col = col
            break
    
    if elev_col:
        print(f'   Elevation column: {elev_col}')
        print(f'   Elevation range: {merged[elev_col].min():.2f} - {merged[elev_col].max():.2f} ft')
        print(f'   Elevation std: {merged[elev_col].std():.2f} ft')
    
    # Save merged data
    output_file = f'{base_dir}/images_and_data.csv'
    merged.to_csv(output_file, index=False)
    print(f'\n   Saved to {output_file}')
    
    return merged


def main():
    print('='*60)
    print('USGS Camera Data Collection')
    print('='*60)
    
    # Collect from a few known good cameras
    cameras_to_try = [
        ('WI_Yahara_River_at_McFarland_GATES', '05430500'),
        ('MN_Mississippi_River_Abv_37th_Ave_NE_in_Fridley', '05288500'),
        ('PA_Allegheny_River_at_Franklin', '03025500'),
        ('MD_Deer_Creek_at_Eden_Mill_Dam_near_Pylesville', '01579905'),
        ('OR_Klamath_River_below_John_C_Boyle_Powerplant_near_Keno', '11510700'),
        ('WI_Green_Bay_Oil_Depot', '040851385'),
        ('NY_Cannonsville_Reservoir_Diversion_Channel_near_Grahamsville', '01365100'),
        ('NJ_Middle_Brook_at_Burnt_Mills', '01399100'),
        ('WI_Bark_River_near_Rome', '05426250'),
        ('AK_Bradly_River_near_Tidewater_near_Homer', '15239070'),
        ('VA_OPEQUON_CREEK_NEAR_BERRYVILLE': '01615000'),
        ('CO_White_River_below_Boise_Creek_near_Rangely': '09306290'),
        ('DC_Watts_Branch_at_Washington': '01651800'),
        ('WA_Pend_Oreille_River_at_Newport': '12395500'),
        ('ID_Lemhi_River_below_L5_Diversion_near_Salmon': '13305310'),
        ('PA_Swatara_Creek_near_Palmyra': '01573208'),
        ('TX_Elm_Fk_Trinity_Rv_nr_Lewisville': '08053000'),
        ('NY_West_Branch_Croton_River_near_Croton_Falls': '01374701'),
        ('LA_Buxton_Creek_at_Hwy_27_near_DeQuincy': '08016910'),
        ('TX_New_Year_Ck_at_FM_1155_nr_Chappel_Hill': '08111110'),
        ('SC_Congaree_River_below_Cayce_DOWNSTREAM_CAMERA': '021695075'),
        ('CA_Threemile_Slough_Nr_Rio_Vista_CA': '11337080'),
        ('LA_Black_River_at_Jonesville': '07373267'),
        ('SC_Lake_Moultrie_Tailrace_Canal_at_Moncks_Corner': '02172002'),
        ('NJ_Stony_Brook_at_Princeton': '01401000'),
        ('PA_Swatara_Creek_near_Palmyra': '01573208'),
        ('NJ_Great_Egg_Harbor_River_at_Folsom': '01411000'),
        ('CA_Middle_River_at_Middle_River': '11312676'),
        ('ID_Big_Wood_River_at_Hailey_Total_Flow': '13139510'),
        ('KY_OHIO_R_US_OF_MCALPINE_DAM_AT_RRB_AT_LOUISVILLE': '03293551'),
        ('VA_FLATLICK_BRANCH_ABOVE_FROG_BRANCH_AT_CHANTILLY': '01656903'),
        ('OH_Maumee_River_near_Defiance': '04192500'),
        ('UT_Weber_River_at_Gateway_UTAH': '10136500')
    ]
    
    all_data = []
    
    for cam_id, site_id in cameras_to_try:
        try:
            df = collect_data_for_camera(cam_id, site_id, num_images=150)
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f'\n✓ Successfully collected {len(df)} samples from {cam_id}')
            else:
                print(f'\n✗ No data collected from {cam_id}')
        except Exception as e:
            print(f'\n✗ Error with {cam_id}: {e}')
    
    # Combine all data
    if all_data:
        import pandas as pd
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv('data/combined_dataset.csv', index=False)
        print(f'\n{"="*60}')
        print(f'COLLECTION COMPLETE!')
        print(f'{"="*60}')
        print(f'Total samples: {len(combined)}')
        print(f'Saved to: data/combined_dataset.csv')
    else:
        print('\nNo data collected. Try different cameras.')


if __name__ == '__main__':
    main()