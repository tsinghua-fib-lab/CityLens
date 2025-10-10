
import numpy as np
import pandas as pd
import os
import random
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from evaluate.utils import get_response_mllm_api, convert_image_to_webp_base64
from config import STV_NUM, GLOBAL_TASK_NUM
random.seed(42)

def single_task_gen(city, task_name):
    data_path = f''
    sat_stv_path = f""
    stv_folder = f""
    sat_folder = f""
    task_path = f''
    if not os.path.exists(os.path.dirname(task_path)):
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
    df_data = pd.read_csv(data_path, dtype={'img_name': str})
    df_stv = pd.read_csv(sat_stv_path, dtype={'file_name': str, 'source_zip': str})

    task_map = {
        "gdp": "GDP",
        "pop": "worldpop",
        "acc2health": "walk_to_health",
        "carbon": "carbon",
        "build_height": "build_height",
    }
    task_indicator_map = {
        "gdp": "average GDP(the unit is PPP 2005 international dollars)",
        "pop": "total population",
        "acc2health": "average walking-only travel time to healthcare",
        "carbon": "total carbon emissions monthly",
        "build_height": "average of the net building height in meters"
    }
    indicator = task_indicator_map[task_name]

    prompt = f"Suppose you are a professional socioeconomic data analyst in {city}. Based on the provided satellite image and several street view photos taken within the same area covered by the satellite image, please estimate 'the {indicator}' for this area. Consider factors such as building structures, road infrastructure, visible traffic patterns, land use characteristics, greenery, and any other relevant features."
    all_data = []
    for _, row in df_data.iterrows():
        img_name = row['img_name']
        reference_value = row[task_map[task_name]]

        if pd.isna(reference_value):
            continue
        expected_zip = f"{city}{img_name}.zip"
        matched_stv = df_stv[df_stv['source_zip'] == expected_zip]

        if matched_stv.empty:
            continue

        stv_paths_all = [os.path.join(stv_folder, os.path.basename(path)) for path in matched_stv['file_name'].tolist()]

        if len(stv_paths_all) > STV_NUM:
            stv_paths = random.sample(stv_paths_all, STV_NUM)
        else:
            stv_paths = stv_paths_all
        if len(stv_paths) < STV_NUM:
            continue

        sat_path_full = os.path.join(sat_folder, img_name + ".png")

        images = [sat_path_full] + stv_paths

        all_data.append({
            'area': img_name,
            'images': images,
            'prompt': prompt,
            'reference': reference_value
        })
    with open(task_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print("len(all_data)", len(all_data))

def single_task_gen_china(city, task_name):
    data_path = f''
    sat_stv_path = f''
    task_path = f''
    if not os.path.exists(os.path.dirname(task_path)):
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
    df = pd.read_csv(data_path, dtype={'img_name': str})
    with open(sat_stv_path, 'r') as f:
        sat_stv = json.load(f)
    sat_prefix = f''
    stv_prefix = f""
    all_data = []
    task_map = {
        "gdp": "GDP",
        "pop": "worldpop",
        "acc2health": "walk_to_health",
        "carbon": "carbon",
        "build_height": "build_height",
    }
    task_indicator_map = {
        "gdp": "average GDP(the unit is PPP 2005 international dollars)",
        "pop": "total population",
        "acc2health": "average walking-only travel time to healthcare",
        "carbon": "total carbon emissions monthly",
        "build_height": "average of the net building height in meters"
    }
    indicator = task_indicator_map[task_name]

    prompt = f"Suppose you are a professional socioeconomic data analyst in {city}. Based on the provided satellite image and several street view photos taken within the same area covered by the satellite image, please estimate 'the {indicator}' for this area. Consider factors such as building structures, road infrastructure, visible traffic patterns, land use characteristics, greenery, and any other relevant features."
    for _, row in df.iterrows():
        y_x = row['img_name']  
        if y_x not in sat_stv:
            continue
        reference_value = row[task_map[task_name]]

        if pd.isna(reference_value):
            continue
        sat_path_full = os.path.join(sat_prefix, y_x + ".png")

        stv_paths_all = [os.path.join(stv_prefix, fname) for fname in sat_stv[y_x]]
        if len(stv_paths_all) > STV_NUM:
            stv_paths = random.sample(stv_paths_all, STV_NUM)
        else:
            stv_paths = stv_paths_all
        if len(stv_paths) < STV_NUM:
            continue
        images = [sat_path_full] + stv_paths

        all_data.append({
            'img_name': y_x,
            'images': images,
            'prompt': prompt,
            'reference': reference_value
        })

    with open(task_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    

def data_gen_simple(city):
    task_names = ["gdp", "pop", "acc2health", "carbon", "build_height"]
    if city == "Beijing" or city == "Shanghai":
        for task_name in task_names:
            single_task_gen_china(city, task_name)
    else:
        for task_name in task_names:
            single_task_gen(city, task_name)

def generate_session_simple(city, data, task_name, prompt_type, model_name):
    url_file = ""
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    task_indicator_map = {
        "gdp": "average GDP(the unit is PPP 2005 international dollars)",
        "pop": "total population",
        "acc2health": "average walking-only travel time to healthcare",
        "carbon": "total carbon emissions monthly",
        "build_height": "average of the net building height in meters"
    }
    indicator = task_indicator_map[task_name]
    example_map = {
        "gdp": 1914240768,
        "pop": 16170,
        "acc2health": 10,
        "carbon": 720,
        "build_height": 12
    }
    example_num = example_map[task_name]
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
            
        if i == STV_NUM:
            break
    if prompt_type == "simple":
        content.append({
            "type": "text",
            "text": f"Please provide a single specific number (not a range or approximate value) for '{indicator}'. No explanation is needed. Example answer: {example_num}\n Answer: "
        })
    elif prompt_type == "normalized":
        content.append({
            "type": "text",
            "text": f"Please provide a single specific number for '{indicator}' (on a scale from 0.0 to 9.9). No explanation is needed. Example answer: 8.8\n Answer: "
        })

    session = [{
        "role": "user",
        "content": content
    }]
    return session

def single_eval_task(args):
    city, model_name, d, prompt_type, task_name = args
    if prompt_type == "simple" or prompt_type == "normalized":
        session = generate_session_simple(city, d, task_name, prompt_type, model_name)
    elif prompt_type == "map":
        session = generate_session_map(city, d, task_name)
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

def eval_task(city, model_name, num_process, prompt_type, task_name):
    if city == "all":
        task_path = f""
    else:
        task_path = f''
    
    model_name_full = model_name.replace("/", "_")
    response_path = f''
    
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(task_path, "r") as f:
        data = json.load(f)
    if len(data) > GLOBAL_TASK_NUM:
        data = random.sample(data, GLOBAL_TASK_NUM)
        
    args_list = [(city, model_name, d, prompt_type, task_name) for d in data]
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
    parser.add_argument('--task_name', type=str, default='gdp', choices=['gdp', "pop", "acc2health", "carbon", "build_height"], help='task name')
    parser.add_argument('--mode', type=str, default='gen', help='gen or eval')
    parser.add_argument('--num_process', type=int, default=10, help='number of processes for evaluation')
    parser.add_argument('--prompt_type', type=str, default='simple', help='simple, map, normalized')
    args = parser.parse_args() 
    print(args)
    
    if args.mode == 'gen':
        print("Generate the data")
        data_gen_simple(args.city_name)
    elif args.mode == 'eval':
        eval_task(args.city_name, args.model_name, args.num_process, args.prompt_type, args.task_name)
    
if __name__ == "__main__":
    main()
