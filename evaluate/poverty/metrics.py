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
    value_str = str(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", value_str)
    if match:
        return float(match.group())
    else:
        print(f"Warning: No valid float found in '{value_str}'")
        return None

def process_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    true_vals = []
    pred_vals = []

    for item in data:
        true_val = extract_float(item['reference'])
        pred_val = extract_float(item['response'])

        true_vals.append(true_val)
        pred_vals.append(pred_val)

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
    parser.add_argument('--task_name', type=str, default='median', choices=['median', 'line_100', 'line_200'], help='task name')
    parser.add_argument("--prompt_type", default="simple", help="Prompt type")

    args = parser.parse_args()
    model_name_full = args.model_name.replace("/", "_")
    response_path = f'/data5/liutianhui/UrbanSensing/results/poverty/{args.task_name}/{args.city_name}/{model_name_full}_{args.prompt_type}_response.json'
    summary_path = f'/data5/liutianhui/UrbanSensing/results/poverty/{args.task_name}_summary.csv'
    y_true, y_pred = process_json(response_path)
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No valid data found.")
    else:
        mse, mae, r2, rmse = compute_regression_metrics(y_pred, y_true)
        print("r2:", r2)
        write_csv(summary_path, args.city_name, args.model_name, args.prompt_type, mse, mae, r2, rmse)
        print(f"Metrics written for city={args.city_name}, model={args.model_name} to {summary_path}")
