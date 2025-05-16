
import numpy as np
import pandas as pd
import os
import random
import json
import argparse
# from pycitydata.map import Map
from tqdm import tqdm
from multiprocessing import Pool
from evaluate.utils import get_response_mllm_api, convert_image_to_webp_base64
from config import STV_NUM, HEALTH_US_NUM
random.seed(42)


### prompt提供图片，城市，地理位置，地理知识
def single_task_gen(city, task_name):
    data_path = f'/data5/liutianhui/UrbanSensing/data/uvi/uvi_data_{city}.csv'
    ct_sat_stv_path = f'/data5/liutianhui/UrbanSensing/data/US_image/sat/{city}_ct_sat_stv.json'
    task_path = f'/data5/liutianhui/UrbanSensing/data/health/{city}/{city}_health_{task_name}_task.json'
    if not os.path.exists(os.path.dirname(task_path)):
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
    df = pd.read_csv(data_path, dtype={'GEOID_ct': str})
    with open(ct_sat_stv_path, 'r') as f:
        ct_sat_stv = json.load(f)
    # print("len(ct_sat_stv)", len(ct_sat_stv))
    prefix = f'/data5/liutianhui/UrbanSensing/data/US_image/sat/{city}/'
    task_map = {
        "cancer": "cancercrud",
        "diabetes": "diabetescr",
        "lpa": "lpacrudepr",
        "mental": "mhlthcrude",
        "physical": "phlthcrude",
        "obesity": "obesitycru",
    }
    task_indicator_map = {
        "cancer": "crude prevalence of cancer (excluding skin cancer) among adults",
        "diabetes": "crude prevalence of diagnosed diabetes among adults",
        "lpa": "crude prevalence of no leisure-time physical activity among adults",
        "mental": "crude prevalence of mental health not good for ≥ 14 days among adults",
        "physical": "crude prevalence of physical health not good for ≥ 14 days among adults",
        "obesity": "crude prevalence of obesity among adults",
    }
    indicator = task_indicator_map[task_name]
    prompt = f"Suppose you are a professional health data analyst in {city}, United States. Based on the provided satellite imagery and several street view photos, please estimate 'the {indicator}' in the census tract where these images are taken. Consider factors such as local healthcare facilities, infrastructure for physical activities, neighborhood conditions, and any other relevant details."
    all_data = []
    for _, row in df.iterrows():
        ct = row['GEOID_ct']
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
            # print(f"ct {ct} 的街景图片数量不足，使用全部 {len(stv_paths_all)} 张")
            stv_paths = stv_paths_all

        # 最后组成image列表
        images = [sat_path_full] + stv_paths
        row_task = task_map[task_name]
        if not pd.isna(row[row_task]):
            all_data.append({
            'ct': ct,
            'images': images,
            'prompt': prompt,
            'reference': row[row_task],
        })
    with open(task_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"数据生成完成：{task_name}")
    print("len(all_data)", len(all_data))

def data_gen_simple(city):
    task_names = ["cancer", "mental", "obesity"]
    for task_name in task_names:
        single_task_gen(city, task_name)
    
    
    
def generate_session_simple(city, data, task_name, prompt_type, model_name):
    """
    根据输入的数据生成交替的文本和图片（url）结构，并返回prompt数据
    """
    url_file = "/data5/liutianhui/ossutil-2.0.3/url_mapping_urbansensing_20250506_1month.csv"
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    task_indicator_map = {
        "cancer": "crude prevalence of cancer (excluding skin cancer) among adults",
        "diabetes": "crude prevalence of diagnosed diabetes among adults",
        "lpa": "crude prevalence of no leisure-time physical activity among adults",
        "mental": "crude prevalence of mental health not good for ≥ 14 days among adults",
        "physical": "crude prevalence of physical health not good for ≥ 14 days among adults",
        "obesity": "crude prevalence of obesity among adults",
    }
    indicator = task_indicator_map[task_name]
    prompt = data['prompt']
    
    content = []
    
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
            # 对于后续图片，添加 street view image {i} 的描述，并插入其 url
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
            "text": f"Please provide a single specific number (not a range or approximate value) for '{indicator}'. No explanation is needed. Example answer: 13.8\n Answer: "
        })
    elif prompt_type == "normalized":
        # if "gemini" in model_name:
        #     print("here")
        #     content.append({
        #         "type": "text",
        #         "text": f"I understand it may be difficult to predict precisely, but please provide your best estimate as a single specific number for '{indicator}' (on a scale from 0.0 to 9.9). No explanation is needed.\n Please answer: "
        #     })
        # else:
        content.append({
            "type": "text",
            "text": f"Please provide a single specific number for '{indicator}' (on a scale from 0.0 to 9.9). No explanation is needed. Example answer: 8.8\n Answer: "
        })

    session = [{
        "role": "user",
        "content": content
    }]
    return session

def generate_session_map(city, data, task_name):
    """
    在prompt中加入地理知识
    """
    geoinfo_csv_path = f'/data5/liutianhui/UrbanSensing/data/crime/{city}_uvi_image_address_place.csv'
    df_geoinfo = pd.read_csv(geoinfo_csv_path)
    task_indicator_map = {
        "cancer": "crude prevalence of cancer (excluding skin cancer) among adults",
        "diabetes": "crude prevalence of diagnosed diabetes among adults",
        "lpa": "crude prevalence of no leisure-time physical activity among adults",
        "mental": "crude prevalence of mental health not good for ≥ 14 days among adults",
        "physical": "crude prevalence of physical health not good for ≥ 14 days among adults",
        "obesity": "crude prevalence of obesity among adults",
    }
    indicator = task_indicator_map[task_name]
    prompt = f"Suppose you are a professional health data analyst in {city}, United States. Based on the provided satellite imagery and several street view photos(with their corresponding addresses and nearby places), please estimate 'the {indicator}' in the census tract where these images are taken. Consider factors such as local healthcare facilities, infrastructure for physical activities, neighborhood conditions, and any other relevant details."
   
    image_info_dict = {}
    for _, row in df_geoinfo.iterrows():
        image = row['image']
        address = row['address']
        nearby_pois = eval(row['nearby_pois'])  # 将字符串转换为实际的列表结构
        image_info_dict[image] = {
            "address": address,
            "nearby_pois": nearby_pois
        }
    content = []
    
    # 先添加文本部分
    content.append({
        "type": "text",
        "text": prompt
    })
    
    images = data.get("images", [])
    
    for i, image_path in enumerate(images):
        assert os.path.exists(image_path), f"Image {image_path} not found"

        # 生成图片的 Base64 编码
        base64_image = convert_image_to_webp_base64(image_path)
        
        if i == 0:
            # 对于第一张图片，添加 satellite image 的描述，并插入其 Base64 编码
            content.append({
                "type": "text",
                "text": "Satellite image: "
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            
        else:
            # 后续图片是street view image
            image_info = image_info_dict[image_path]
            address = image_info['address']
            nearby_pois = image_info['nearby_pois']
            geo_info = f"Address: {address}\nNearby places: "
            # 格式化附近 POI 信息
            nearby_places_info = []
            for poi in nearby_pois:
                poi_name, poi_distance = poi
                nearby_places_info.append(f"{poi_distance} km, {poi_name}")
            geo_info += "\n".join(nearby_places_info)

            content.append({
                "type": "text",
                "text": f"Street view image {i}: "
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            content.append({
                "type": "text",
                "text": f"Street view image {i} geoinformation: \n{geo_info}"
            })
            
    content.append({
        "type": "text",
        "text": f"Please provide a single, exact number for '{indicator}' only (not a range). No explanation is needed.\n Example answer: 13.8"
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

def eval_task(city, model_name, num_process, prompt_type, task_name):
    if city == "US":
        task_path = f"/data5/liutianhui/UrbanSensing/data/health/US_health_{task_name}_task.json"
    else:
        task_path = f'/data5/liutianhui/UrbanSensing/data/health/{city}/{city}_health_{task_name}_task.json'
    
    
    model_name_full = model_name.replace("/", "_")
    response_path = f'/data5/liutianhui/UrbanSensing/results/health/{task_name}/{city}/{model_name_full}_{prompt_type}_response.json'
    
    
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(task_path, "r") as f:
        data = json.load(f)
    if len(data) > HEALTH_US_NUM:
        data = random.sample(data, HEALTH_US_NUM)
    # if args.data_name == "mini":
    #     data = data[:10]
    args_list = [(city, model_name, d, prompt_type, task_name) for d in data]
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
    parser.add_argument('--task_name', type=str, default='physical', choices=['cancer', 'diabetes', 'lpa', 'mental', 'physical', 'obesity'], help='task name')
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