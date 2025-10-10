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
tiff_path = ""
gdal.AllRegister()
walk_dataset = gdal.Open(tiff_path)
walk_transform = walk_dataset.GetGeoTransform()
walk_band = walk_dataset.GetRasterBand(1)
proj = walk_dataset.GetProjection()
print("Projection string (WKT):\n", proj)
srs = osr.SpatialReference()
srs.ImportFromWkt(proj)
if srs.IsProjected:
    print("Projected coordinate system:", srs.GetAttrValue('projcs'))
elif srs.IsGeographic:
    print("Geographic coordinate system:", srs.GetAttrValue('geogcs'))
epsg_code = srs.GetAttrValue('AUTHORITY',1)
print("EPSG code:", epsg_code)


