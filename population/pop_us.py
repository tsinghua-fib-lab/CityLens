import os
import pandas as pd
import json
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from evaluate.utils import get_response_mllm_api, convert_image_to_webp_base64
random.seed(42)
def generate_session_simple(city, data):
    prompt = f"Suppose you are a professional urban data analyst in {city}, United States. Based on the provided street view image, please estimate 'the population density' of the area where the image is taken. Consider factors such as the type of neighborhood(residential, commercial, industrial), the visible infrastructure, and any other relevant features."
    # print("data", data)
    img = data["img_name"]
    image_path = f"/data5/liutianhui/UrbanSensing/data/pop/{city}_CUT/" + img
    assert os.path.exists(image_path), f"Image {image_path} not found"
    content = []
    content.append({
        "type": "text",
        "text": prompt
    })
    base64_image = convert_image_to_webp_base64(image_path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    content.append({
        "type": "text",
        "text": "Please provide a single, exact 'population density' number only (not a range). No explanation is needed.\n Example answer: 14738.933"
    })

    session = [{
        "role": "user",
        "content": content
    }]
    return session

def generate_session_map(city, data):
    geoinfo_csv_path = f'/data5/liutianhui/UrbanSensing/data/pop/{city}_image_address_place.csv'
    df_geoinfo = pd.read_csv(geoinfo_csv_path)
    prompt = f"Suppose you are a professional urban data analyst in {city}, United States. Based on the provided street view image, please estimate 'the population density' of the area where the image is taken. Consider factors such as the type of neighborhood(residential, commercial, industrial), the visible infrastructure, and any other relevant features."
    
    img = data["img_name"]
    image_path = f"/data5/liutianhui/UrbanSensing/data/pop/{city}_CUT/" + img
    assert os.path.exists(image_path), f"Image {image_path} not found"
    image_info_dict = {}
    for _, row in df_geoinfo.iterrows():
        image = row['image']
        address = row['address']
        nearby_pois = eval(row['nearby_pois'])  # 将字符串转换为实际的列表结构
        image_info_dict[image] = {
            "address": address,
            "nearby_pois": nearby_pois
        }
    image_info = image_info_dict[img]
    address = image_info['address']
    nearby_pois = image_info['nearby_pois']
    geo_info = f"Address: {address}\nNearby places: "
    # 格式化附近 POI 信息
    nearby_places_info = []
    for poi in nearby_pois:
        poi_name, poi_distance = poi
        nearby_places_info.append(f"{poi_distance} km, {poi_name}")
    geo_info += "\n".join(nearby_places_info)
    content = []
    content.append({
        "type": "text",
        "text": prompt
    })
    base64_image = convert_image_to_webp_base64(image_path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    content.append({
                "type": "text",
                "text": f"Street view image geoinformation: \n{geo_info}"
            })
    content.append({
        "type": "text",
        "text": "Please provide a single, exact 'population density' number only (not a range). No explanation is needed.\n Example answer: 14738.933"
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
    # print(session)
    reference = d["worldpop"]
    img_path = d["img_name"]

    ret = get_response_mllm_api(session, model_name, temperature=0, max_tokens=2000, infer_server=None,json_mode=False)
    return {
        "images": img_path,
        "session": session,
        "reference": reference,
        "response": ret
    }

def eval_task(city, model_name, num_process, prompt_type):
    task_path = f'/data5/liutianhui/UrbanSensing/data/pop/{city}_data.csv'
    model_name_full = model_name.replace("/", "_")
    response_path = f'/data5/liutianhui/UrbanSensing/results/pop/{city}/{model_name_full}_{prompt_type}_pop_response.json'
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(task_path)
    ## 去除异常值
    # df = df[df['worldpop'] >= 1000]
    data = df.to_dict(orient='records')
    if len(data) >= 50:
        sampled_data = random.sample(data, 50)
    else:
        sampled_data = data

    # if args.data_name == "mini":
    #     data = data[:10]
    args_list = [(city, model_name, d, prompt_type) for d in sampled_data]
    response = []
    with Pool(num_process) as pool:
        print("Processing tasks in parallel...")
        # 使用 imap 处理每个任务
        for result in tqdm(pool.imap(single_eval_task, args_list), total=len(sampled_data)):
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
    parser.add_argument('--prompt_type', type=str, default='simple', help='simple, map, normalized')
    args = parser.parse_args() 
    print(args)
    
    eval_task(args.city_name, args.model_name, args.num_process, args.prompt_type)
    
if __name__ == "__main__":
    main()