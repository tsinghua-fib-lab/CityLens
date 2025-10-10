from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import os
import argparse
from utils_scp import *

SAVE_DIR = ''
SAVE_PATH = Path(SAVE_DIR)
SAVE_PATH.mkdir(exist_ok=True, parents=True)

MULTI_PROCESS_NUM = 4
SAMPLE_SIZE = 30       
time_limit = 300       

def crawl_from_csv(csv_file_path, index=[]):
    df = pd.read_csv(csv_file_path)
    grouped = df.groupby("region_code")

   

    pool = Pool(MULTI_PROCESS_NUM)
    results = []

    for i, (region_code, group) in enumerate(grouped):
        if index and i not in index: 
            continue

        lat_list = group["latitude"].tolist()
        lon_list = group["longitude"].tolist()

        results.append(
            pool.apply_async(
                crawl_single_region_csv_order,
                args=(i, lat_list, lon_list, region_code)
            )
        )

    for r in results:
        r.get()
    pool.close()
    pool.join()


def crawl_single_region_csv_order(index, lat_list, lon_list, region_code):
    path_with_region_code = SAVE_PATH.joinpath(region_code)
    path_with_region_code.mkdir(exist_ok=True, parents=True)

    detected_points = 0
    success_points = 0
    total_start_time = datetime.now()

    for lat, lon in zip(lat_list, lon_list):
        detected_points += 1
        lat = np.round(float(lat), 6)
        lon = np.round(float(lon), 6)

        try:
            panoid, real_lat, real_lon, date = get_metadata_from_lati_lonti(lat, lon)
        except Exception as e:

            continue

        if panoid in [-1, -2, -3]:

            continue

        img_dict = get_image_tiles(panoid)
        save_img(img_dict, path_with_region_code, panoid, lat, lon, real_lat, real_lon, date)

        success_points += 1
        elapsed = (datetime.now() - total_start_time).total_seconds()


        if success_points >= SAMPLE_SIZE:
            break

        if elapsed > time_limit:

            break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest="start", type=int, default=0, help=' index')
    parser.add_argument('--end', dest="end", type=int, default=None, help=' index')
    args = parser.parse_args()
    print(args)

    csv_path = ''

    begin_time = datetime.now()
    df = pd.read_csv(csv_path)
    grouped_len = len(df.groupby("region_code"))

    if args.end is None or args.end > grouped_len:
        index_range = [i for i in range(args.start, grouped_len)]
    else:
        index_range = [i for i in range(args.start, args.end)]

    crawl_from_csv(csv_path, index=index_range)
    end_time = datetime.now()

    print( str((end_time - begin_time).total_seconds() / 3600) )
