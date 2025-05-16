import json
import csv

# 输入输出文件路径
model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
model_name_full = model_name.replace("/", "_")
json_file = f"/data5/liutianhui/UrbanSensing/data/feature/{model_name_full}/health_mental_streetview_images_response.json"  # 替换为你的实际路径
output_csv = f"/data5/liutianhui/UrbanSensing/data/feature/{model_name_full}/health_mental_streetview_feature.csv"

# 13个固定的指标顺序
indicators = [
    "Person", "Bike", "Heavy Vehicle", "Light Vehicle", "Façade", "Window & Opening",
    "Road", "Sidewalk", "Street Furniture", "Greenery - Tree",
    "Greenery - Grass & Shrubs", "Sky", "Nature"
]

# 读取 JSON 文件
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理并写入 CSV 文件
with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    # 写入表头
    writer.writerow(["image_name"] + indicators)

    for image_name, text in data.items():
        # 将字符串按行分割，每行格式为 “指标: 分数”
        lines = text.strip().split("\n")
        values = {line.split(":")[0].strip(): float(line.split(":")[1].strip()) for line in lines if ":" in line}
        # 保证顺序写入每个指标值（如缺失则写None）
        row = [image_name] + [values.get(ind, None) for ind in indicators]
        writer.writerow(row)

print(f"已保存到 {output_csv}")
