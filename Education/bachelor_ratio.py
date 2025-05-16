import numpy as np
import pandas as pd
import os
import random
import json
import argparse
# from pycitydata.map import Map
from tqdm import tqdm
from multiprocessing import Pool
from evaluate.urban_utils import get_response_mllm_api, convert_image_to_webp_base64
from config import STV_NUM, EDUCATION_US_NUM
random.seed(42)


### prompt提供图片，城市，地理位置，地理知识
def data_gen_simple(city):
    data_path = f'/data5/panghetian/UrbanSensing/uvi_data_{city}.csv'
    ct_sat_stv_path = f'/data5/liutianhui/UrbanSensing/data/US_image/sat/{city}_ct_sat_stv.json'
    task_path = f'/data5/panghetian/UrbanSensing/{city}_bachelor_ratio_task.json'
    df = pd.read_csv(data_path, dtype={'census_tract': str})
    with open(ct_sat_stv_path, 'r') as f:
        ct_sat_stv = json.load(f)
    prefix = f'/data5/liutianhui/UrbanSensing/data/US_image/sat/{city}/'
    prompt = f"Suppose you are a demographic analyst specializing in education statistics in {city}, United States. Based on the provided satellite imagery and several street view photos, please estimate bachelor ratio in the census tract where the images are taken. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details."
    all_data = []


    for _, row in df.iterrows():
        ct = row['census_tract']  # 确保 key 是字符串
        if ct not in ct_sat_stv:
            # print(f"ct {ct} 不在 JSON 里，跳过")
            continue

        sat_path = ct_sat_stv[ct]['sat_path']
        stv_paths_all = ct_sat_stv[ct]['stv_paths']

        # 拼接 sat_path 完整路径
        sat_path_full = os.path.join(prefix, sat_path) + ".png"

        # 如果超过STV_NUM张，随机选STV_NUM张；否则用全部
        if len(stv_paths_all) > STV_NUM:
            stv_paths = random.sample(stv_paths_all, STV_NUM)
        else:
            print(f"ct {ct} 的街景图片数量不足，使用全部 {len(stv_paths_all)} 张")
            stv_paths = stv_paths_all

        # 最后组成image列表
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
    print(f"已保存到 {task_path}")
    print(f"数据生成完成，共 {len(all_data)} 条数据")
    
def generate_session_simple(city, data, prompt_type, model_name):
    """
    根据输入的数据生成交替的文本和图片（url）结构，并返回prompt数据
    """
    url_file = f"/data5/liutianhui/ossutil-2.0.3/url_mapping_urbansensing_20250506_1month.csv"
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    
    prompt = data['prompt']
    content = []
    
    # 先添加文本部分
    content.append({
        "type": "text",
        "text": prompt
    })
    
    images = data.get("images", [])
    
    for i, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        # 查找对应 url
        image_url = url_dict.get(image_name)
        if not image_url:
            raise ValueError(f"❌ 没有在 mapping 文件中找到 {image_name} 的 URL")
        
        if i == 0:
            # 对于第一张图片，添加 satellite image 的描述，并插入其 Base64 编码
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
            # 对于后续图片，添加 street view image {i} 的描述，并插入其 Base64 编码
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
    # elif prompt_type == "map":
    #     session = generate_session_map(city, d)
    # print(session)
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
    task_path = f'/data5/panghetian/UrbanSensing/{city}_bachelor_ratio_task.json'
    model_name_full = model_name.replace("/", "_")
    response_path = f'/data5/panghetian/UrbanSensing/results/bachelor_ratio/{city}/{model_name_full}_{prompt_type}_bachelor_ratio_response.json'
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(task_path, "r") as f:
        data = json.load(f)
    if len(data) > EDUCATION_US_NUM:
        data = random.sample(data, EDUCATION_US_NUM) 
    # if args.data_name == "mini":
    #     data = data[:10]
    args_list = [(city, model_name, d, prompt_type) for d in data]
    response = []
    with Pool(num_process) as pool:
        print("Processing tasks in parallel...")
        # 使用 imap 处理每个任务
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