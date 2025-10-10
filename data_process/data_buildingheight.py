import pandas as pd
import numpy as np
import math
from math import floor
from osgeo import gdal
from pyproj import Transformer
def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg
from pyproj import Transformer

transformer = Transformer.from_crs("", "", always_xy=True)

tiff_path = ""
gdal.AllRegister()
build_height_dataset = gdal.Open(tiff_path)
build_height_transform = build_height_dataset.GetGeoTransform()
build_height_band = build_height_dataset.GetRasterBand(1)
def get_build_height_value(tile_name):
    x = int(tile_name.split('_')[1].split('.')[0])
    y = int(tile_name.split('_')[0])
    lat_max,lng_min=num2deg(x,y,zoom=15)
    _,lng_max=num2deg(x+1,y,zoom=15)
    lat_min,_=num2deg(x,y+1,zoom=15)
    lng_min_54009, lat_max_54009 = transformer.transform(lng_min, lat_max)
    lng_max_54009, lat_min_54009 = transformer.transform(lng_max, lat_min)
    y_init = int(floor((lat_max_54009 - build_height_transform[3]) / build_height_transform[5]))
    x_init = int(floor((lng_min_54009 - build_height_transform[0]) / build_height_transform[1]))
    x_end = int(floor((lng_max_54009 - build_height_transform[0]) / build_height_transform[1]))
    y_end = int(floor((lat_min_54009 - build_height_transform[3]) / build_height_transform[5]))
    data = build_height_band.ReadAsArray(x_init, y_init, x_end - x_init, y_end - y_init)
    if data.size == 0:
        return np.nan
    data[data < 0] = 0
    return np.array(data.mean()).tolist() 
cities = ["Beijing", "Shanghai", "CapeTown", "London", "NewYork", "Paris", "SanFrancisco", "Moscow", "Tokyo", "Mumbai", "Nairobi", "SaoPaulo", "Sydney"]
for city in cities:
    csv_path = f""
    df = pd.read_csv(csv_path)
    build_height_values = []
    for img_name in df['img_name']:
        try:
            val = get_build_height_value(img_name)
        except Exception as e:
            val = np.nan
        build_height_values.append(val)
    df['build_height_to_health'] = build_height_values
    df.to_csv(csv_path, index=False)
