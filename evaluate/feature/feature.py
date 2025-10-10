import json
import csv
import os
import pandas as pd


model_name = ""
model_name_full = model_name.replace("/", "_")
task_file = "" 
single_feature_file = f"" 
output_file = f""

feature_df = pd.read_csv(single_feature_file)
feature_df.set_index("image_name", inplace=True)

with open(task_file, 'r', encoding='utf-8') as f:
    task_data = json.load(f)
indicator_cols = [
    "Person", "Bike", "Heavy Vehicle", "Light Vehicle", "Façade", "Window & Opening",
    "Road", "Sidewalk", "Street Furniture", "Greenery - Tree",
    "Greenery - Grass & Shrubs", "Sky", "Nature"
]

output_rows = []

for item in task_data:
    ct = item["ct"]
    images = item["images"]
    reference = item.get("reference")
    reference_normalized = item.get("reference_normalized")
    streetview_images = [os.path.basename(p) for p in images[1:]]

    feature_vectors = []
    for img in streetview_images:
        if img in feature_df.index:
            feature_vectors.append(feature_df.loc[img][indicator_cols].astype(float).tolist())

    if feature_vectors:
        avg_vector = pd.DataFrame(feature_vectors, columns=indicator_cols).mean().tolist()
        row = [ct] + avg_vector + [reference, reference_normalized]
        output_rows.append(row)

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ["ct"] + indicator_cols + ["reference", "reference_normalized"]
    writer.writerow(header)
    writer.writerows(output_rows)

