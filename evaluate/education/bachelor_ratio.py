import numpy as np
import pandas as pd
import os
import random
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from evaluate.urban_utils import get_response_mllm_api, convert_image_to_webp_base64
from config import STV_NUM, EDUCATION_US_NUM
random.seed(42)

def data_gen_simple(city):
    data_path = f''
    ct_sat_stv_path = f''
    task_path = f''
    df = pd.read_csv(data_path, dtype={'census_tract': str})
    with open(ct_sat_stv_path, 'r') as f:
        ct_sat_stv = json.load(f)
    prefix = f''
    prompt = f""
    all_data = []


    for _, row in df.iterrows():
        ct = row['census_tract']  
        if ct not in ct_sat_stv:
            continue

        sat_path = ct_sat_stv[ct]['sat_path']
        stv_paths_all = ct_sat_stv[ct]['stv_paths']

        sat_path_full = os.path.join(prefix, sat_path) + ".png"

        if len(stv_paths_all) > STV_NUM:
            stv_paths = random.sample(stv_paths_all, STV_NUM)
        else:
            stv_paths = stv_paths_all

        images = [sat_path_full] + stv_paths
        reference = row['bachelor_ratio']
        if not pd.isna(reference):
            all_data.append({
                'ct': ct,
                'images': images,
                'prompt': prompt,
                'reference': reference
            })

    with open(task_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    
def generate_session_simple(city, data, prompt_type, model_name):
    url_file = f""
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    
    prompt = data['prompt']
    content = []
    
    content.append({
        "type": "text",
        "text": prompt
    })
    
    images = data.get("images", [])
    
    for i, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        image_url = url_dict.get(image_name)
        
        if i == 0:
            content.append({
                "type": "text",
                "text": "Satellite image: "
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
            
        else:
            content.append({
                "type": "text",
                "text": f"Street view image {i}: "
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
            
        if i == STV_NUM:
            break
    if prompt_type == "simple":
        content.append({
            "type": "text",
            "text": "Please provide a single specific bachelor ratio number (not a range or approximate value), expressed as a decimal between 0 and 1. No explanation is needed. Example answer: 0.46.\n Answer:"
        })
    elif prompt_type == "normalized":
        content.append({
            "type": "text",
            "text": f"Please provide a single specific number for bachelor ratio level (on a scale from 0.0 to 9.9). No explanation is needed. Example answer: 8.8.\n Answer:"
        })

    session = [{
        "role": "user",
        "content": content
    }]
    return session

def single_eval_task(args):
    city, model_name, d, prompt_type = args
    if prompt_type == "simple"or prompt_type == "normalized":
        session = generate_session_simple(city, d, prompt_type, model_name)
    reference = d["reference"]
    reference_normalized = d["reference_normalized"]
    img_path = d["images"]

    ret = get_response_mllm_api(session, model_name, temperature=0, max_tokens=2000, infer_server=None,json_mode=False)
    return {
        "images": img_path,
        "session": session,
        "reference": reference,
        "reference_normalized": reference_normalized,
        "response": ret
    }

def eval_task(city, model_name, num_process, prompt_type):
    task_path = f''
    model_name_full = model_name.replace("/", "_")
    response_path = f''
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(task_path, "r") as f:
        data = json.load(f)
    if len(data) > EDUCATION_US_NUM:
        data = random.sample(data, EDUCATION_US_NUM) 
    args_list = [(city, model_name, d, prompt_type) for d in data]
    response = []
    with Pool(num_process) as pool:
        print("Processing tasks in parallel...")
        for result in tqdm(pool.imap(single_eval_task, args_list), total=len(data)):
            if result:
                response.append(result)

    with open(response_path, "w") as f:
        json.dump(response, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='NewYork', help='city name')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-vl2', help='model name')
    parser.add_argument('--mode', type=str, default='gen', help='gen or eval')
    parser.add_argument('--num_process', type=int, default=10, help='number of processes for evaluation')
    parser.add_argument('--prompt_type', type=str, default='simple', help='simple, normalized')
    args = parser.parse_args() 
    print(args)
    
    if args.mode == 'gen':
        print("Generate the data")
        data_gen_simple(args.city_name)
    elif args.mode == 'eval':
        eval_task(args.city_name, args.model_name, args.num_process, args.prompt_type)
    
if __name__ == "__main__":
    main()
