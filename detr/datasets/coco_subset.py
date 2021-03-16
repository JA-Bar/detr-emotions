import json
from glob import glob
from pathlib import Path

from detr.datasets.data_utils import convert_bboxes_format

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoSubset(Dataset):
    def __init__(self,
                 coco_path,
                 target_classes,
                 transforms=None,
                 mode='train',
                 val_split=0.1):
        super().__init__()

        coco_path = Path(coco_path)
        self.coco_path = coco_path
        self.transforms = transforms

        assert coco_path.exists() and coco_path.is_dir(), "Path to coco is not the base directory"

        data, idx_to_classes, classes_to_idx = self.parse_annotations(coco_path, target_classes)

        # The split is always going to be deterministic, the first n go to train,
        # while the last m go to val, better implementation pending
        n_data = len(data)
        n_split = int(n_data * (1-val_split))
        if mode == 'train':
            data = list(data.values())[:n_split]
        else:
            data = list(data.values())[n_split:]

        self.image_data = [d[0] for d in data]
        self.annotation_data = [d[1:] for d in data]
        self.idx_to_classes = idx_to_classes
        self.classes_to_idx = classes_to_idx

    def __getitem__(self, idx):
        image_data = self.image_data[idx]
        ann_data = self.annotation_data[idx]

        image = Image.open(self.coco_path/'images'/image_data['file_name'])
        class_labels = [ann['category_id'] for ann in ann_data]
        bboxes = [ann['bbox'] for ann in ann_data]

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image),
                                          bboxes=bboxes,
                                          class_labels=class_labels)
            image = transformed['image']
            class_labels = transformed['class_labels']
            bboxes = transformed['bboxes']

            image_h, image_w = image.shape[-2:]
            bboxes = convert_bboxes_format(bboxes, 'coco', 'yolo', image_h, image_w)

            class_labels = torch.tensor(class_labels, dtype=torch.long)
            bboxes = torch.tensor(bboxes, dtype=torch.float)

        labels = {'bboxes': bboxes, 'classes': class_labels}
        return image, labels

    def __len__(self):
        return len(self.image_data)

    @staticmethod
    def parse_annotations(coco_path, target_classes):
        annotations_file = glob(str(coco_path/'annotations'/'instances*.json'))[0]

        with open(annotations_file) as f:
            annotations = json.load(f)

        idx_to_classes = {cat['id']: cat['name'] for cat in annotations['categories']}
        classes_to_idx = {name: idx for (idx, name) in idx_to_classes.items()}

        saved_images = glob(str(coco_path/'images'/'*.jpg'))
        saved_images = set(img.split('/')[-1] for img in saved_images)

        target_classes = set(classes_to_idx[cat] for cat in target_classes)

        annotation_data = {img['id']: [img] for img in annotations['images']
                           if img['file_name'] in saved_images}

        for ann in annotations['annotations']:
            img_id = ann['image_id']
            class_id = ann['category_id']
            if img_id in annotation_data and class_id in target_classes:
                annotation_data[img_id].append(ann)

        return annotation_data, idx_to_classes, classes_to_idx

