import os
import argparse
import numpy as np
import pandas as pd
import random
import math
import base64
import subprocess

from pycitydata.map import Map
from citysim.routing import RoutingClient
from config import MAP_CACHE_PATH, RESOURCE_PATH, RESULTS_PATH, MONGODB_URI, MAP_DICT, PROXY, IMAGE_FOLDER 

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r

def extract_coords_from_filename(city, image_filename):
    meta_file = os.path.join(IMAGE_FOLDER, f"")
    parts = image_filename.split('_')
    dataset_name = parts[0]
    sid_84_long = parts[1]
    sid_84_lat = parts[2]
    sid = parts[3].split('.')[0]  

    df = pd.read_csv(meta_file)

    matched_row = df[(df['sid_84_long'] == float(sid_84_long)) & 
                     (df['sid_84_lat'] == float(sid_84_lat)) & 
                     (df['sid'] == sid)]

    return matched_row.iloc[0]['longitude_origin'], matched_row.iloc[0]['latitude_origin']





    



