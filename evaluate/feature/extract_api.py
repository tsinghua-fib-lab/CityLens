import json
import csv

model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
model_name_full = model_name.replace("/", "_")
json_file = f""  
output_csv = f""

indicators = [
    "Person", "Bike", "Heavy Vehicle", "Light Vehicle", "Façade", "Window & Opening",
    "Road", "Sidewalk", "Street Furniture", "Greenery - Tree",
    "Greenery - Grass & Shrubs", "Sky", "Nature"
]

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)

    writer.writerow(["image_name"] + indicators)

    for image_name, text in data.items():

        lines = text.strip().split("\n")
        values = {line.split(":")[0].strip(): float(line.split(":")[1].strip()) for line in lines if ":" in line}

        row = [image_name] + [values.get(ind, None) for ind in indicators]
        writer.writerow(row)

