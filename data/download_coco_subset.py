import argparse
from glob import glob
import os
import zipfile
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm
import requests


def download_annotations(coco_path, file_name='coco_annotations'):
    COCO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    annotations_path = (coco_path/file_name).with_suffix('.zip')
    description = f'Downloading annotations to {str(annotations_path)}'

    annotations = requests.get(COCO_URL, allow_redirects=True, stream=True)
    file_fize = int(annotations.headers.get('content-length'))

    with open(annotations_path, 'wb') as f:

        with tqdm(total=file_fize, unit='B', unit_scale=True,
                  desc=description, initial=0) as pbar:

            for ch in annotations.iter_content(chunk_size=1024):
                if ch:
                    f.write(ch)
                    pbar.update(len(ch))


def unzip_annotations(coco_path, file_name='coco_annotations', keep_files=False):
    print('Unzipping annotations file...')
    annotations_path = (coco_path/file_name).with_suffix('.zip')
    with zipfile.ZipFile(annotations_path, 'r') as f:
        f.extractall(coco_path)

    # clean up everything but the validation (smallest) file
    if not keep_files:
        all_files = set(glob(str(coco_path/'annotations/*.json')))
        annotations_file = set(glob(str(coco_path/'annotations/instances_val*.json')))
        files_to_clean = all_files - annotations_file

        confirm = input(f"Delete files: {files_to_clean} as part of the cleanup [y/n]? ")
        if 'y' in confirm:
            for file in files_to_clean:
                os.remove(file)

            os.remove(annotations_path)
            print('Cleanup done!')


def download_images(categories, coco_path, limit=10):
    print(f'Downloading images for categories {categories}')
    base_images_path = coco_path/'images'
    if not base_images_path.exists():
        base_images_path.mkdir(parents=True)

    annotations_path = str(coco_path/'annotations'/'instances_val*.json')
    coco = COCO(glob(annotations_path)[0])

    images_data = []
    for cat in categories:
        cat_ids = coco.getCatIds(catNms=[cat])
        img_ids = coco.getImgIds(catIds=cat_ids)
        images = coco.loadImgs(img_ids)
        images_data.extend(images[:min(len(images), limit)])

    for im in tqdm(images_data, desc='Downloading'):
        im_data = requests.get(im['coco_url'])
        with open(base_images_path/im['file_name'], 'wb') as f:
            f.write(im_data.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='')
    parser.add_argument('--categories', default='truck, boat')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--keep_files', action='store_true')
    args = parser.parse_args()

    save_path = args.save_path
    if not args.save_path:
        save_path = Path(__file__).parent/'coco'
    else:
        save_path = Path(save_path)

    if not (save_path/'annotations').exists():
        download_annotations(save_path)
        unzip_annotations(save_path, keep_files=args.keep_files)

    categories = args.categories.replace(', ', ',').split(',')

    download_images(categories, save_path, args.limit)
    print('Coco subset downloaded!')

