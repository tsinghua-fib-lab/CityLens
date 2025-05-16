import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import json
def num2deg(x, y, zoom=15):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat_deg = np.rad2deg(lat_rad)
    return lon_deg, lat_deg

def deg2num(lon_deg, lat_deg, zoom=15):
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    return xtile, ytile



city = "Beijing"  # 选择城市
# 目录路径
image_folder = f'/data5/liutianhui/UrbanSensing/data/street_view/{city}_CUT' 
rs_dataset_csv = f'/data5/liutianhui/UrbanSensing/data/remote_sensing/{city}_img_indicators.csv'
rs_dataset_df = pd.read_csv(rs_dataset_csv)
rs_data_set = set(rs_dataset_df['img_name'].tolist())

si_sv_mapping_data = {}

# 匹配两种文件名中的经纬度
pattern1 = re.compile(r'.*&([0-9.]+)&([0-9.]+)&.*\.jpg$')  # 匹配格式1
pattern2 = re.compile(r'_([0-9.]+)_([0-9.]+)_')            # 匹配格式2



# 遍历图像文件夹
for filename in tqdm(os.listdir(image_folder)):
    try:
        filepath = os.path.join(image_folder, filename)
        if not filename.lower().endswith('.jpg'):
            continue

        lat = lng = None

        # 尝试第一种格式
        m1 = pattern1.match(filename)
        if m1:
            lng = float(m1.group(1))
            lat = float(m1.group(2))
        else:
            # 尝试第二种格式
            m2 = pattern2.search(filename)
            if m2:
                lng = float(m2.group(1))
                lat = float(m2.group(2))

        if lat is None or lng is None:
            print(f"跳过未识别格式：{filename}")
            continue

        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            print(f"跳过非法经纬度：{filename} lat={lat}, lng={lng}")
            continue

        zoom = 15
        x, y = deg2num(lng, lat, zoom)
        si_img_name = f'{y}_{x}'

        if si_img_name in rs_data_set:
            if si_img_name not in si_sv_mapping_data:
                si_sv_mapping_data[si_img_name] = []
            si_sv_mapping_data[si_img_name].append(filename)

    except Exception as e:
        print(f"处理文件 {filename} 出错: {e}")
        continue

# 保存匹配结果
with open(f'/data5/liutianhui/UrbanSensing/data/global_image/{city}_sat_stv_mapping.json', 'w') as f:
    json.dump(si_sv_mapping_data, f, indent=2)

# 统计信息
unique_si_images = len(si_sv_mapping_data)
total_sv_images = sum(len(set(v)) for v in si_sv_mapping_data.values())

print(f"\n统计信息：")
print(f"匹配到的 si_img_name 数量（遥感图像）: {unique_si_images}")
print(f"总共关联的 img_name 数量（街景图像，去重后）: {total_sv_images}")
si_with_10_or_more_sv = sum(1 for v in si_sv_mapping_data.values() if len(v) >= 2)

print(f"街景图像数量 >= 2 的 si_img_name 数量（遥感图像）: {si_with_10_or_more_sv}")