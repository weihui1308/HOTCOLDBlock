import json
import shutil
import cv2
import pandas as pd
from PIL import Image

from utils import *


def convert_coco_json(use_segments=False):
    save_dir = make_dirs('path/')
    jsons = ["home/coco.json"]
    coco80 = coco91_to_coco80_class()
    
    fn = Path(save_dir) / 'labels'
    for json_file in sorted(jsons):
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            if x['iscrowd'] or (x['category_id'] != 1):
                continue

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name'][5:]
            
            if x['bbox'][3] <= 120:
                continue
            
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            
            # Segments
            segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
            s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
            
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                line = coco80[x['category_id'] - 1], *(s if use_segments else box)  # cls, box or segments
                with open((fn / f).with_suffix('.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
            
                shutil.copy("path1"+img['file_name'], 'path/images/')


if __name__ == '__main__':
    convert_coco_json()