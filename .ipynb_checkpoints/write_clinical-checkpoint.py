import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    print(captions)
    print(iid)

    return [
        binary,
        captions,
        str(iid),
    ]


def make_arrow(dataset_root='arrows'):
    with open(f"/home/azureuser/BLIP/medical_data/medical_20k_segment_blip.json", "r") as fp:
        captions = json.load(fp)
    
    img_dir = '/home/azureuser/BLIP/medical_data/medical_data_20k/'

    caption_paths = {}
    for cap in captions:
        img = cap['image_id']
        img_path = img_dir + str(img) + '.png'
        if img_path not in caption_paths:
            caption_paths[img_path] = []
        caption_paths[img_path].append(cap['caption'])

    bs = [path2rest(path, caption_paths[path]) for path in tqdm(caption_paths)]
    dataframe = pd.DataFrame(
        bs, columns=["image", "caption", "image_id"],
    )
    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/clinical.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

make_arrow()
