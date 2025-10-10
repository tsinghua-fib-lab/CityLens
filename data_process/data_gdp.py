import pandas as pd
import numpy as np
import math
from math import floor
from osgeo import gdal
from osgeo import gdal, osr
def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg
tiff_path =""
gdal.AllRegister()
gdp_dataset = gdal.Open(tiff_path)
gdp_transform = gdp_dataset.GetGeoTransform()
gdp_band = gdp_dataset.GetRasterBand(1)
proj = gdp_dataset.GetProjection()
print("Projection string (WKT):\n", proj)
srs = osr.SpatialReference()
srs.ImportFromWkt(proj)
if srs.IsProjected:
    print("Projected coordinate system:", srs.GetAttrValue('projcs'))
elif srs.IsGeographic:
    print("Geographic coordinate system:", srs.GetAttrValue('geogcs'))
epsg_code = srs.GetAttrValue('AUTHORITY',1)
print("EPSG code:", epsg_code)
def get_gdp_value(tile_name):
    x = int(tile_name.split('_')[1].split('.')[0])
    y = int(tile_name.split('_')[0])
    lat_max,lng_min=num2deg(x,y,zoom=15)
    _,lng_max=num2deg(x+1,y,zoom=15)
    lat_min,_=num2deg(x,y+1,zoom=15)
    y_init = int(floor((lat_max - gdp_transform[3]) / gdp_transform[5]))
    x_init = int(floor((lng_min - gdp_transform[0]) / gdp_transform[1]))
    x_end = int(floor((lng_max - gdp_transform[0]) / gdp_transform[1]))
    y_end = int(floor((lat_min - gdp_transform[3]) / gdp_transform[5]))
    data = gdp_band.ReadAsArray(x_init, y_init, x_end - x_init, y_end - y_init)
    if data.size == 0:
        return np.nan
    data[data < 0] = 0
    return np.array(data.mean()).tolist() 
cities = ["Beijing", "Shanghai", "CapeTown", "London", "NewYork", "Paris", "SanFrancisco", "Moscow", "Tokyo", "Mumbai", "Nairobi", "SaoPaulo", "Sydney"]
for city in cities:
    csv_path = f""
    df = pd.read_csv(csv_path)
    gdp_values = []
    for img_name in df['img_name']:
        try:
            val = get_gdp_value(img_name)
        except Exception as e:
            val = np.nan
        gdp_values.append(val)
    df['GDP'] = gdp_values
    df.to_csv(csv_path, index=False)

