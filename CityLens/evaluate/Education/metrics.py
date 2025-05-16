import json
import csv
import argparse
import os
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(pred_list, true_list):
    mse = mean_squared_error(true_list, pred_list)
    mae = mean_absolute_error(true_list, pred_list)
    r2 = r2_score(true_list, pred_list)
    rmse = mean_squared_error(true_list, pred_list, squared=False)
    return mse, mae, r2, rmse

def extract_float(value):
    if isinstance(value, (float, int)):
        return float(value)

    # 2️⃣ 如果是字符串
    if isinstance(value, str):
        # 2.1 如果字符串太长，直接跳过
        if len(value) > 100:
            return None

        # 2.2 如果字符串中有非数字字符（除了小数点、逗号）
        if not re.match(r"^[\d\.,]+$", value.strip()):
            # 有其他字符 → 认为无效
            return None

        # 2.3 替换逗号
        value_str = value.replace(",", "")

        try:
            return float(value_str)
        except ValueError:
            return None

    # 3️⃣ 其他情况（比如 None、list 等）直接返回 None
    return None


def process_json(json_path, prompt_type):
    with open(json_path, 'r') as f:
        data = json.load(f)

    true_vals = []
    pred_vals = []

    for item in data:
        if prompt_type == "simple":
            true_val = extract_float(item['reference'])
        elif prompt_type == "normalized":
            true_val = extract_float(item['reference_normalized'])
        pred_val = extract_float(item['response'])
        if true_val is None or pred_val is None:
            print(f"Warning: Invalid data found in item: {item['response']}")
            continue

        true_vals.append(true_val)
        pred_vals.append(pred_val)
    print("length of vals: ", len(true_vals), len(pred_vals))
    return true_vals, pred_vals

def write_csv(output_path, city, model, prompt_type, mse, mae, r2, rmse):
    fieldnames = ['city', 'model', "prompt_type", 'MSE', 'MAE', 'R2', 'RMSE']
    write_header = not os.path.exists(output_path)

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'city': city,
            'model': model,
            'prompt_type': prompt_type,
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': rmse
        })

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", default="NewYork", help="City name")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-32B-Instruct", help="Model name")
    parser.add_argument('--task_name', type=str, default='physical', choices=['cancer', 'diabetes', 'lpa', 'mental', 'physical', 'obesity'], help='task name')
    parser.add_argument("--prompt_type", default="simple", help="Prompt type")

    args = parser.parse_args()
    model_name_full = args.model_name.replace("/", "_")
    response_path = f'/data5/panghetian/UrbanSensing/results/bachelor_ratio/{args.city_name}/{model_name_full}_{args.prompt_type}_response.json'
    summary_path = f'/data5/panghetian/UrbanSensing/results/bachelor_ratio/US_bachelor_ratio_summary.csv'
    y_true, y_pred = process_json(response_path, args.prompt_type)
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No valid data found.")
    else:
        mse, mae, r2, rmse = compute_regression_metrics(y_pred, y_true)
        print("r2:", r2)
        write_csv(summary_path, args.city_name, args.model_name, args.prompt_type, mse, mae, r2, rmse)
        print(f"Metrics written for city={args.city_name}, model={args.model_name} to {summary_path}")