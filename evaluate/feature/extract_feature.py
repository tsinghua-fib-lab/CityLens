import csv
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from evaluate.utils import get_response_mllm_api

def feature_prompt():
    return """Analyze the provided street view image. For each of the following 13 indicators, provide a score from 0.0 to 9.9 representing its presence or prominence in the image. The output should only be the indicator name followed by its score, one indicator per line.
    Indicators:
        Person
        Bike
        Heavy Vehicle
        Light Vehicle
        Façade
        Window & Opening
        Road
        Sidewalk
        Street Furniture
        Greenery - Tree
        Greenery - Grass & Shrubs
        Sky
        Nature
    Example:
    Person: 2.5
    Bike: 0.0
    ……"""

def process_image(args):
    image_name, image_url, model_name = args
    prompt = feature_prompt()
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    session = [{"role": "user", "content": content}]
    try:
        ret = get_response_mllm_api(session, model_name, temperature=0, max_tokens=2000, infer_server=None, json_mode=False)
        return image_name, ret
    except Exception as e:
        return image_name, None

def main():
    model_name = ""
    image_paths_csv = ""
    model_name_full = model_name.replace("/", "_")
    
    image_info_csv = ""
    output_dir = f""
    response_file_prefix = "all_streetview_images_response_part"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    name_to_url = {}
    with open(image_info_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_to_url[row["image_name"]] = row["image_url"]


    task_list = []
    with open(image_paths_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            image_path = row[0]
            image_name = os.path.basename(image_path)
            image_url = name_to_url.get(image_name)
            if image_url:
                task_list.append((image_name, image_url, model_name))

    save_interval = 1000 
    result_dict = {}
    part_idx = 1
    total_tasks = len(task_list)


    result_dict = {}
    with Pool(processes=20) as pool:
        for idx, (image_name, ret) in enumerate(tqdm(pool.imap_unordered(process_image, task_list, chunksize=1), total=total_tasks), 1):
            if ret is not None:
                result_dict[image_name] = ret
            
            if idx % save_interval == 0 or idx == total_tasks:
                part_response_file = os.path.join(output_dir, f"{response_file_prefix}_{part_idx}.json")
                with open(part_response_file, 'w', encoding='utf-8') as fout:
                    json.dump(result_dict, fout, indent=4, ensure_ascii=False)

                part_idx += 1
                result_dict.clear()


if __name__ == "__main__":
    main()
