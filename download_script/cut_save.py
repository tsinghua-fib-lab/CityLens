from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

IMG_SIZE = 512
CROP_SIZE = 1536
CROP_NUM = 4
OUT_SIZE = 512
NUM_PROCESSES = 30  

def crop_random(img, save_path, panoid, lat, lon, pano_name):
    canvas_w, canvas_h = img.size
    if canvas_w <= CROP_SIZE:
        box = (0, 0, canvas_w, canvas_h)
        crop_index = 0
    else:
        crop_index = np.random.randint(CROP_NUM)
        left = crop_index * CROP_SIZE
        right = min((crop_index + 1) * CROP_SIZE, canvas_w)
        if right <= left:
            left = 0
            right = min(CROP_SIZE, canvas_w)
            crop_index = 0
        box = (left, 0, right, canvas_h)

    cropped = img.crop(box)
    cropped = cropped.resize((OUT_SIZE, OUT_SIZE))
    crop_name = f"{panoid}&{lat}&{lon}&crop_{crop_index}.jpg"
    cropped.save(save_path / crop_name)
    return crop_name, crop_index


def process_single_area(area_path, cut_root):
    area = Path(area_path)
    files = list(area.glob("*.jpg"))
    if not files:
        return

    groups = defaultdict(list)
    pano_info = dict()

    for f in files:
        parts = f.stem.split("&")
        panoid = parts[0]
        lat, lon = parts[1], parts[2]
        x, y = int(parts[-2]), int(parts[-1])
        groups[panoid].append((f, x, y))
        pano_info[panoid] = (lat, lon)

    save_cut_dir = Path(cut_root) / area.name
    save_cut_dir.mkdir(parents=True, exist_ok=True)

    cut_info_list = []

    for panoid, imgs in tqdm(groups.items(), desc=f"Processing {area.name}", unit="pano"):
        xs = [x for _, x, _ in imgs]
        ys = [y for _, _, y in imgs]
        x_max = max(xs)
        y_min, y_max = min(ys), max(ys)

        if x_max == 12:
            total_cols = 13
        elif x_max == 15:
            total_cols = 16
        else:
            continue
        canvas = Image.new("RGB", (IMG_SIZE * total_cols, IMG_SIZE * (y_max - y_min + 1)))
        for f, x, y in imgs:
            try:
                img_tile = Image.open(f)
                canvas.paste(img_tile, (x * IMG_SIZE, (y - y_min) * IMG_SIZE))
            except Exception:
                continue

        lat, lon = pano_info[panoid]
        pano_name = f"{panoid}&{lat}&{lon}.jpg"

        crop_name, crop_index = crop_random(canvas, save_cut_dir, panoid, lat, lon, pano_name)
        cut_info_list.append({
            'crop_file_name': crop_name,
            'panoid': panoid,
            'lat': lat,
            'lon': lon,
            'crop_index': crop_index,
            'original_pano': pano_name
        })

    pd.DataFrame(cut_info_list).to_csv(save_cut_dir / "cut_meta_info.csv", index=False)


def process_area_wrapper(area_path):
    return process_single_area(area_path, cut_root_global)


def main():
    global cut_root_global

    cut_root = ""
    input_root = ""

    cut_root_global = cut_root

    Path(cut_root).mkdir(parents=True, exist_ok=True)

    areas = [str(p) for p in Path(input_root).iterdir() if p.is_dir()]

    with Pool(processes=NUM_PROCESSES) as pool:
        list(tqdm(pool.imap(process_area_wrapper, areas),
                  total=len(areas)))

if __name__ == "__main__":
    main()
