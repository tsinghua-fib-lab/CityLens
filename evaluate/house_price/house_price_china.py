
import numpy as np
import pandas as pd
import os
import random
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from evaluate.utils import get_response_mllm_api, convert_image_to_webp_base64
from config import STV_NUM, HOUSE_PRICE_US_NUM
random.seed(42)

def data_gen_simple(city):
    data_path = f''
    sat_stv_path = f''
    task_path = f''
    if not os.path.exists(os.path.dirname(task_path)):
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
    df = pd.read_csv(data_path, dtype={'y_x': str})
    with open(sat_stv_path, 'r') as f:
        sat_stv = json.load(f)
    sat_prefix = f''
    stv_prefix = f""
    prompt = f"Suppose you are a professional real estate appraisal expert in {city}, China. Based on the provided satellite image and several street view photos taken within the same area covered by the satellite image, please estimate 'the average house price(in yuan/m²)' for this area. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details."

    all_data = []


    for _, row in df.iterrows():
        y_x = row['y_x']  
        if y_x not in sat_stv:
            continue

        sat_path_full = os.path.join(sat_prefix, y_x + ".png")
        stv_paths_all = [os.path.join(stv_prefix, fname) for fname in sat_stv[y_x]]
        if len(stv_paths_all) > STV_NUM:
            stv_paths = random.sample(stv_paths_all, STV_NUM)
        else:
            stv_paths = stv_paths_all
        if len(stv_paths) < 10:
            continue
        images = [sat_path_full] + stv_paths
        reference = row['price']
        if not pd.isna(reference):
            all_data.append({
                'y_x': y_x,
                'images': images,
                'prompt': prompt,
                'reference': reference
            })

    with open(task_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

def generate_session_simple(city, data):
    url_file = f""
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    prompt = f"Suppose you are a professional real estate appraisal expert in {city}, China. Based on the provided satellite image and several street view photos taken within the same area covered by the satellite image, please estimate 'the average house price(in yuan/m²)' for this area. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details."
    
    content = []
    content.append({
        "type": "text",
        "text": prompt
    })
    
    images = data.get("images", [])
    
    for i, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        image_url = url_dict.get(image_name)
        if not image_url:
            raise ValueError
        
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
            
    content.append({
        "type": "text",
        "text": "Please provide a single, exact house price number only (not a range). No explanation is needed.\n Example answer: 40766"
    })

    session = [{
        "role": "user",
        "content": content
    }]
    return session


def single_eval_task(args):
    city, model_name, d, prompt_type = args
    if prompt_type == "simple":
        session = generate_session_simple(city, d)
    elif prompt_type == "map":
        session = generate_session_map(city, d)
    reference = d["reference"]
    img_path = d["images"]

    ret = get_response_mllm_api(session, model_name, temperature=0, max_tokens=2000, infer_server=None,json_mode=False)
    return {
        "images": img_path,
        "session": session,
        "reference": reference,
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
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-vl2', help='model name')
    parser.add_argument('--mode', type=str, default='gen', help='gen or eval')
    parser.add_argument('--num_process', type=int, default=10, help='number of processes for evaluation')
    parser.add_argument('--prompt_type', type=str, default='simple', help='simple, map, normalized')
    args = parser.parse_args() 
    print(args)
    
    if args.mode == 'gen':
        print("Generate the data")
        data_gen_simple(args.city_name)
    elif args.mode == 'eval':
        eval_task(args.city_name, args.model_name, args.num_process, args.prompt_type)
    
if __name__ == "__main__":
    main()
