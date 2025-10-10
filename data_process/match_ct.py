import geopandas as gpd
import shapely

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon
from tqdm import tqdm
output_path = 'all_img_set_SanFrancisco.json'
all_county = gpd.read_file('cb_2019_us_tract_500k.shp')
target_geoids = pd.read_csv(
    "2020_Census_Tracts_sanfrancisco.csv",
    dtype={"geoid": str}
)["geoid"].tolist()
all_county["STATEFP"]=all_county["STATEFP"].apply(lambda x:str(x).zfill(3))


all_county=all_county[all_county["STATEFP"]!="002"]
all_county=all_county[all_county["STATEFP"]!="015"]
all_county=all_county[all_county["STATEFP"]!="060"]
all_county=all_county[all_county["STATEFP"]!="069"]
all_county=all_county[all_county["STATEFP"]!="072"]
all_county=all_county[all_county["STATEFP"]!="078"]
all_county=all_county[all_county["STATEFP"]!="066"]
all_county["STATE_COUNTY"]  = all_county["STATEFP"].apply(lambda x:str(x).zfill(3)) + all_county["COUNTYFP"].apply(lambda x:str(x).zfill(3))
print(len(set(all_county["STATE_COUNTY"].unique())))
sample_county = set(all_county["STATE_COUNTY"].unique())

import random 
sample_1000_county_gpd = all_county[all_county["STATE_COUNTY"].isin(sample_county)]
sample_1000_county_gpd = sample_1000_county_gpd[sample_1000_county_gpd["GEOID"].astype(str).isin(target_geoids)]
sample_1000_county_gpd = sample_1000_county_gpd.reset_index(drop=True)
sample_1000_county_gpd["index"] = sample_1000_county_gpd.index

print(len(sample_1000_county_gpd))
print(sample_1000_county_gpd["GEOID"].unique())

import warnings
warnings.filterwarnings("ignore")
def num2deg(x, y, zoom=15):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat_deg = np.rad2deg(lat_rad)
    return lat_deg, lon_deg

def deg2num(lon_deg, lat_deg, zoom=15):
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    return xtile, ytile

def compute_tile_coordinates(min_x, max_x, min_y, max_y):
    x_arr = np.arange(min_x, max_x + 1)
    y_arr = np.arange(min_y, max_y + 1)
    lon_arr, lat_arr = num2deg_batch(x_arr, y_arr)
    return lon_arr, lat_arr,x_arr, y_arr

def num2deg_batch(x_arr, y_arr, zoom=15):
    n = 2.0 ** zoom
    lon_deg_arr = x_arr / n * 360.0 - 180.0
    lat_rad_arr = np.arctan(np.sinh(np.pi * (1 - 2 * y_arr / n)))
    lat_deg_arr = np.rad2deg(lat_rad_arr)
    return lon_deg_arr, lat_deg_arr

def create_tile_polygons(lon_arr, lat_arr,x_arr, y_arr):
    
    tile_gpd= gpd.GeoDataFrame()
    lon_mesh, lat_mesh = np.meshgrid(lon_arr, lat_arr, indexing='ij')
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')
    
    
    vertices = np.array([
        lon_mesh[:-1, :-1], lat_mesh[:-1, :-1],
        lon_mesh[1:, :-1], lat_mesh[1:, :-1],
        lon_mesh[1:, 1:], lat_mesh[1:, 1:],
        lon_mesh[:-1, 1:], lat_mesh[:-1, 1:]
    ])

    vertices = vertices.reshape(4, 2, -1)
    vertices = np.transpose(vertices, axes=(2, 0, 1))
    polygons = [Polygon(p) for p in vertices]
    vertices_x_y = np.array([
        x_mesh[:-1, :-1], y_mesh[:-1, :-1],
        x_mesh[1:, :-1], y_mesh[1:, :-1],
        x_mesh[1:, 1:], y_mesh[1:, 1:],
        x_mesh[:-1, 1:], y_mesh[:-1, 1:]
    ])
    
    vertices_x_y = vertices_x_y.reshape(4, 2, -1)
    vertices_x_y = np.transpose(vertices_x_y, axes=(2, 0, 1))
    y_x = [f"{int(p[0][1])}_{int(p[0][0])}" for p in vertices_x_y]
    tile_gpd['geometry'] = polygons
    tile_gpd['y_x'] = y_x
    return tile_gpd


def process_county(one_index):
    temp_shp = sample_1000_county_gpd[sample_1000_county_gpd.index==one_index]
    max_y=deg2num(min(temp_shp.bounds.minx),min(temp_shp.bounds.miny))[1]+5
    min_y=deg2num(max(temp_shp.bounds.maxx),max(temp_shp.bounds.maxy))[1]-5
    max_x=deg2num(max(temp_shp.bounds.maxx),max(temp_shp.bounds.maxy))[0]+5
    min_x=deg2num(min(temp_shp.bounds.minx),min(temp_shp.bounds.miny))[0]-5

    temp_y_x=[]
    lon_arr, lat_arr,x_arr, y_arr = compute_tile_coordinates(min_x, max_x, min_y, max_y)
    one_street_shp_x_y = create_tile_polygons(lon_arr, lat_arr,x_arr, y_arr)
    one_street_shp_x_y["tile_area"] = one_street_shp_x_y["geometry"].area

    overlay_gdf = one_street_shp_x_y.overlay(temp_shp, how='intersection')
    overlay_gdf['intersection_area'] = overlay_gdf['geometry'].area
    overlay_gdf['intersection_ratio'] = overlay_gdf['intersection_area'] / overlay_gdf['tile_area']
    overlay_gdf = overlay_gdf[overlay_gdf['intersection_ratio'] > 0.85]

    temp_y_x = overlay_gdf.y_x
    temp_y_x = list(set(temp_y_x))

    geoid = sample_1000_county_gpd.loc[one_index,'GEOID']
    

    return temp_y_x,str(geoid)

from multiprocessing import Pool
import time

import collections
num_proc = 30
t1 = time.time()
pool = Pool(processes=num_proc)
fail_img_list=[]
all_img_list=[]
to_do_list = list(sample_1000_county_gpd.index)
all_img_set = {}
all_img_gpd = gpd.GeoDataFrame()
with tqdm(total=len(to_do_list)) as t:
    for result in pool.imap(process_county, to_do_list):
        if len(result[0])>0:
            all_img_set[result[1]] = result[0]
        t.update()
print("all_img_set",len(all_img_set))

import json

with open(output_path, 'w') as f:
    json.dump(all_img_set, f, indent=4)

