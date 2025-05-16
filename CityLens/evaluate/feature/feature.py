import json
import csv
import os
import pandas as pd

# 文件路径
model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
model_name_full = model_name.replace("/", "_")
task_file = "/data5/liutianhui/UrbanSensing/data/health/US_health_mental_task.json"  # 替换为你的 JSON 文件
single_feature_file = f"/data5/liutianhui/UrbanSensing/data/feature/{model_name_full}/health_mental_streetview_feature.csv"  # 替换为你的 CSV 文件
output_file = f"/data5/liutianhui/UrbanSensing/data/health/US_health_mental_mlp_task_{model_name_full}.csv"

# 读取图像特征 CSV
feature_df = pd.read_csv(single_feature_file)
feature_df.set_index("image_name", inplace=True)

# 读取 JSON 数据
with open(task_file, 'r', encoding='utf-8') as f:
    task_data = json.load(f)

# 指标列
indicator_cols = [
    "Person", "Bike", "Heavy Vehicle", "Light Vehicle", "Façade", "Window & Opening",
    "Road", "Sidewalk", "Street Furniture", "Greenery - Tree",
    "Greenery - Grass & Shrubs", "Sky", "Nature"
]

# 准备输出结果
output_rows = []

for item in task_data:
    ct = item["ct"]
    images = item["images"]
    reference = item.get("reference")
    reference_normalized = item.get("reference_normalized")

    # 跳过第一个卫星图
    streetview_images = [os.path.basename(p) for p in images[1:]]

    # 收集每张图像的特征
    feature_vectors = []
    for img in streetview_images:
        if img in feature_df.index:
            feature_vectors.append(feature_df.loc[img][indicator_cols].astype(float).tolist())
        else:
            print(f"警告：找不到图像特征：{img}")

    if feature_vectors:
        # 对每个指标求平均
        avg_vector = pd.DataFrame(feature_vectors, columns=indicator_cols).mean().tolist()
        row = [ct] + avg_vector + [reference, reference_normalized]
        output_rows.append(row)
    else:
        print(f"跳过 ct={ct}：无有效图像特征")

# 写入输出 CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ["ct"] + indicator_cols + ["reference", "reference_normalized"]
    writer.writerow(header)
    writer.writerows(output_rows)

print(f"结果已保存到 {output_file}")
