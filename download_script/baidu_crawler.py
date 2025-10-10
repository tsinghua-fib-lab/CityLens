import numpy as np
import os
from pathlib import Path
import time
from urllib.request import urlopen

import cv2
import json
import pandas as pd
from coordTransform_utils import bd09_to_wgs84
import ast
import argparse

from config import SAMPLE_POINT_PATH, IMAGE_FOLDER

def coord_conv(long,latit):
    try:
        url = 'http://api.map.baidu.com/geoconv/v1/?coords=' + str(long) + ',' + str(latit) + '&from=5&to=6&ak={}'.format(baidu_key)
        conn = urlopen(url)
        out = conn.read()
        conn.close()
    except:
        time.sleep(600)
        url = 'http://api.map.baidu.com/geoconv/v1/?coords=' + str(long) + ',' + str(latit) + '&from=5&to=6&ak={}'.format(baidu_key)
        conn = urlopen(url)
        out = conn.read()
        conn.close()
    out_json = json.loads(out)
    
    return out_json['result'][0]['x'], out_json['result'][0]['y']

def coord_conv_back(long,latit):
    try:
        url = 'http://api.map.baidu.com/geoconv/v1/?coords=' + str(long) + ',' + str(latit) + '&from=6&to=5&ak={}'.format(baidu_key)
        conn = urlopen(url)
        out = conn.read()
        conn.close()
    except:
        print('坐标转换API无响应')
        time.sleep(600)
        url = 'http://api.map.baidu.com/geoconv/v1/?coords=' + str(long) + ',' + str(latit) + '&from=6&to=5&ak={}'.format(baidu_key)
        conn = urlopen(url)
        out = conn.read()
        conn.close()
    out_json = json.loads(out)
    
    return out_json['result'][0]['x'], out_json['result'][0]['y']

def loc_to_sid(x,y):
    try:
        url = 'https://mapsv0.bdimg.com/?udt=20200902&qt=qsdata&x=' + str(x) + '&y=' + str(y) + '&l=17.031000000000002&action=0&mode=day&t=1530956939770'
        conn = urlopen(url)
        out = conn.read()
        conn.close()
    except:
        time.sleep(600)
        url = 'https://mapsv0.bdimg.com/?udt=20200902&qt=qsdata&x=' + str(x) + '&y=' + str(y) + '&l=17.031000000000002&action=0&mode=day&t=1530956939770'
        conn = urlopen(url)
        out = conn.read()
        conn.close()

    out_json = json.loads(out)
    

    if 'content' in out_json:
        return out_json['content']['id'],out_json['content']['x'],out_json['content']['y']
    else:
        return -1,-1,-1

def sid_to_pic(sid,heading,name):
    try:
        url = 'https://mapsv0.bdimg.com/?qt=pr3d&fovy=50&quality=100&panoid=' + str(sid) + '&heading=' + str(heading) + '&pitch=0&width=1024&height=512'
        conn = urlopen(url)
        print(url)
        outimg = conn.read()
        conn.close()
    except:
        url = 'https://mapsv0.bdimg.com/?qt=pr3d&fovy=50&quality=100&panoid=' + str(sid) + '&heading=' + str(heading) + '&pitch=0&width=1024&height=512'
        conn = urlopen(url)
        outimg = conn.read()
        conn.close()

    if len(outimg) < 10000:
        print('No pic at this point!')
        return 0
    else:
        data_img = cv2.imdecode(np.asarray(bytearray(outimg), dtype=np.uint8), 1)
        conn.close()
        cv2.imwrite(name, data_img) 
        print('Pic Saved!')
        return 1
    
def read_progress(progress_file):
    if Path(progress_file).exists():
        with open(progress_file, 'r') as f:
            progress = int(f.read().strip())
        return progress
    return 0

def save_progress(progress_file, progress):
    with open(progress_file, 'w') as f:
        f.write(str(progress))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Shanghai", choices=["Shanghai", "Beijing"])
    args = parser.parse_args()

    result_path = os.path.join(SAMPLE_POINT_PATH, f'StreetView_Images_{args.city_name}.csv')
    output_dir = os.path.join(IMAGE_FOLDER, f'{args.city_name}_StreetView_Images')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True,parents=True)

    crawled_district = os.listdir(output_dir)
    baidu_key = os.environ.get('BAIDU_KEY')

    csv_path = os.path.join(SAMPLE_POINT_PATH, f'')
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        print("Processing " + str(df['code'][index]))
        dataset_name = str(df['code'][index])

        if dataset_name in crawled_district:
            print(f"{dataset_name} Already crawled, skip!")
            continue

        else:
            try:
                dataset_path = output_dir + dataset_name + "/"
                Path(dataset_path).mkdir(exist_ok=True,parents=True)
                out_csv = dataset_path + ''

                long_array = ast.literal_eval(df['longitude'][index])
                lati_array = ast.literal_eval(df['latitude'][index])
                long_array = [round(x,4) for x in long_array]
                lati_array = [round(x,4) for x in lati_array]

                success_num = 0
                total_num = 0
                no_sid_num = 0
                same_sid_num = 0
                sid_set = set() 
                origin_long_lati_set = set()

                if Path(out_csv).exists():
                    image_info_df = pd.read_csv(out_csv)
                else:
                    image_info_df = pd.DataFrame(columns=['longitude_origin','latitude_origin','longitude_baidu','latitude_baidu','sid','sid_x_baidu','sid_y_baidu','sid_84_long', 'sid_84_lat'])

                for root, dirs, files in os.walk(output_path):
                    for img in files:
                        divide = img.split('_')
                        try:
                            origin_long_lati_set.add((divide[1],divide[2]))
                        except:
                            print(divide)

                progress_file = dataset_path + 'progress.txt'
                start_index = read_progress(progress_file)
                

                for i in range(start_index, len(long_array)):
                    longitude, latitude = long_array[i], lati_array[i]
                    if (str(longitude),str(latitude)) in origin_long_lati_set:

                        continue
                    total_num = total_num + 1


                    try:
                  
                        print(longitude, latitude)
                        long_baidu, latit_baidu = coord_conv(longitude,latitude)
                        time.sleep(3*np.random.random())
                        sid, sid_x, sid_y = loc_to_sid(long_baidu,latit_baidu)
                        sid_x = sid_x / 100
                        sid_y = sid_y / 100
            
                        if sid == -1:
                            print('No sid at this location!')
                            no_sid_num = no_sid_num + 1
                            continue
                        elif sid in sid_set:
                            print('Already has this sid!')
                            same_sid_num = same_sid_num + 1
                            continue
                        else:
                            sid_set.add(sid)
                            success_num = success_num + 1
                            sid_baidu_x, sid_baidu_y = coord_conv_back(sid_x,sid_y)
                            sid_84_long, sid_84_lat = bd09_to_wgs84(sid_baidu_x, sid_baidu_y)
                            for heading in [0,90,180,270]:
                      
                                img_name = dataset_path + dataset_name + "_" + str(sid_84_long)+ "_" + str(sid_84_lat) + "_" + str(sid) + ".jpg"
                                result = sid_to_pic(sid,heading,img_name)
                                if result == 1:
                                    if heading == 270:
                                  
                                        temp_df = pd.DataFrame([[longitude, latitude, long_baidu, latit_baidu, sid, sid_x, sid_y,sid_84_long, sid_84_lat]],columns=['longitude_origin','latitude_origin','longitude_baidu','latitude_baidu','sid','sid_x_baidu','sid_y_baidu','sid_84_long', 'sid_84_lat'])
                                        image_info_df = pd.concat([image_info_df, temp_df], ignore_index=True)
                                        image_info_df.to_csv(out_csv,sep = ',',index = False)
                            save_progress(progress_file, i)
                    except Exception as e:
                        print(f"Error at {i}: {e}")
                        save_progress(progress_file, i)
                        time.sleep(5) 
                        continue


                temp_df = pd.DataFrame([[dataset_name, success_num]], columns=['code', 'success_num'])
                temp_df.to_csv(result_path, mode='a', header=False, index=False)

                print("There are " + str(total_num) + " points")
                print("There are " + str(success_num) + " success points")
                print("There are " + str(same_sid_num) + " points repeated")
                print("There are " + str(no_sid_num) + " points that doesn't have data.")
                image_info_df.to_csv(out_csv,sep = ',',index = False)
            
            except Exception as e:
                print(dataset_name, e)
                continue
