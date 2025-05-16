
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
from config import STV_NUM, HOUSE_PRICE_US_NUM
random.seed(42)


### prompt提供图片，城市，地理位置，地理知识
def data_gen_simple(city):
    data_path = f'/data5/liutianhui/UrbanSensing/data/house_price/China/{city}_house_price_avg.csv'
    sat_stv_path = f'/data5/liutianhui/UrbanSensing/data/street_view/{city}/{city}_sat_stv.json'
    task_path = f'/data5/liutianhui/UrbanSensing/data/house_price/China/{city}_house_price_task.json'
    if not os.path.exists(os.path.dirname(task_path)):
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
    df = pd.read_csv(data_path, dtype={'y_x': str})
    with open(sat_stv_path, 'r') as f:
        sat_stv = json.load(f)
    sat_prefix = f'/data5/liutianhui/UrbanSensing/data/remote_sensing/{city}/'
    stv_prefix = f"/data5/liutianhui/UrbanSensing/data/street_view/{city}/images/"
    prompt = f"Suppose you are a professional real estate appraisal expert in {city}, China. Based on the provided satellite image and several street view photos taken within the same area covered by the satellite image, please estimate 'the average house price(in yuan/m²)' for this area. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details."

    all_data = []


    for _, row in df.iterrows():
        y_x = row['y_x']  # 确保 key 是字符串
        if y_x not in sat_stv:
            # print("y_x not in sat_stv_mapping")
            continue

        sat_path_full = os.path.join(sat_prefix, y_x + ".png")

        # ➤ 为每张街景图像加前缀
        stv_paths_all = [os.path.join(stv_prefix, fname) for fname in sat_stv[y_x]]

        # 如果超过STV_NUM张，随机选STV_NUM张；否则用全部
        if len(stv_paths_all) > STV_NUM:
            stv_paths = random.sample(stv_paths_all, STV_NUM)
        else:
            print(f"y_x {y_x} 的街景图片数量不足，使用全部 {len(stv_paths_all)} 张")
            stv_paths = stv_paths_all
        if len(stv_paths) < 10:
            print(f"y_x {y_x} 的街景图片数量不足，跳过")
            continue

        # 最后组成image列表
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
    print(f"已保存到 {task_path}")
    print(f"数据生成完成，共 {len(all_data)} 条数据")
    

def generate_session_simple(city, data):
    """
    根据输入的数据生成交替的文本和图片（url）结构，并返回prompt数据
    """
    url_file = f"/data5/liutianhui/ossutil-2.0.3/url_mapping_citybench13cities_20250507_1month.csv"
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    prompt = f"Suppose you are a professional real estate appraisal expert in {city}, China. Based on the provided satellite image and several street view photos taken within the same area covered by the satellite image, please estimate 'the average house price(in yuan/m²)' for this area. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details."
    
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
            
    content.append({
        "type": "text",
        "text": "Please provide a single, exact house price number only (not a range). No explanation is needed.\n Example answer: 40766"
    })

    session = [{
        "role": "user",
        "content": content
    }]
    return session

def generate_session_map(city, data):
    """
    在prompt中加入地理知识
    """
    geoinfo_csv_path = f'/data5/liutianhui/UrbanSensing/data/economic/house_price/US/{city}_house_price_task_image_address_place.csv'
    df_geoinfo = pd.read_csv(geoinfo_csv_path)
    prompt = f"Suppose you are a professional real estate appraisal expert in {city}, United States. Based on the provided satellite imagery and several street view photos(with their corresponding addresses and nearby places), please estimate 'the market value of the home' in the census tract where the images are taken. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details."
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
        "text": "Please provide a single, exact home value number only (not a range). No explanation is needed.\n Example answer: 516340"
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
    task_path = f'/data5/liutianhui/UrbanSensing/data/house_price/UK/{city}_house_price_task.json'
    model_name_full = model_name.replace("/", "_")
    response_path = f'/data5/liutianhui/UrbanSensing/results/house_price/{city}/{model_name_full}_{prompt_type}_house_price_response.json'
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(task_path, "r") as f:
        data = json.load(f)
    # if len(data) > HOUSE_PRICE_US_NUM:
    #     data = random.sample(data, HOUSE_PRICE_US_NUM)
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