import os.path

import staintools
import cv2
from pathlib import Path
from tqdm import tqdm

target = staintools.read_image("/Users/jiangxiaofeng/Desktop/Ref.png")
transform_path = Path('/Users/jiangxiaofeng/Desktop/test_img')
img_list = Path(transform_path).glob('**/*.jpg')

normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target)



for img in tqdm(img_list):
    # outpath = str(img).replace('NEW_BLOCKS_MCO','Macenko_MCO')
    outpath = str(img).replace('test_img', 'Macenko_MCO')
    if os.path.exists(outpath):
        print('skipping...')
        continue
    outdir = Path(outpath)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    to_transform = staintools.read_image(str(img))

    transformed = normalizer.transform(to_transform)
    cv2.imwrite(outpath, transformed)
