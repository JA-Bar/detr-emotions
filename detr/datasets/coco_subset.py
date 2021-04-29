import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from detr.utils.box_ops import convert_bboxes_format


class CocoSubset(Dataset):
    def __init__(self,
                 coco_path,
                 target_classes,
                 transforms=None,
                 mode='train',
                 train_val_split=0.8):
        """Dataset built on a subset of coco.

        Read the existing saved images corresponsing to coco, pair them with annotation data,
        and build a dataset including the classes of 'target_classes'.

        Args:
            coco_path (str): Base directory that holds coco annotations and images.
            target_classes (list[str]): Classes to include in the dataset. Their name must
                match the official name defined by the coco dataset.
            transforms (Albumentation Transform): Albumentation transforms that will be applied
                to the images and annotations.
            mode (str): Either 'train' or 'val', define the split that will be used to separate the data.
            train_val_split (str): Number between 0 and 1 that will define the percentage of the data that will
                be used for the 'train' split, the rest (1-train_val_split) will go to 'val'.

        Returns:
            Tuple of (image, labels), where image is a PIL Image with the transformation applied (usually a Tensor),
            and labels is a dictionary with theys:
                'bboxes': Tensor of size [n_objects_in_image, 4] containing the bounding boxes of objects.
                'classes': Tensor of size [n_objects_in_image] containing the corresponsing class id.
        """
        super().__init__()

        coco_path = Path(coco_path)
        self.coco_path = coco_path
        self.transforms = transforms

        assert coco_path.exists() and coco_path.is_dir(), "Path to coco is not the base directory"

        data, idx_to_classes, classes_to_idx = self.parse_annotations(coco_path, target_classes)

        # The split is always going to be deterministic, the first n go to train,
        # while the last m go to val, better implementation pending
        n_data = len(data)
        n_split = int(n_data * (train_val_split))

        data_values = list(data.values())
        random.Random(42).shuffle(data_values)

        if mode == 'train':
            data = data_values[:n_split]
        else:
            data = data_values[n_split:]

        self.image_data = [d[0] for d in data]
        self.annotation_data = [d[1:] for d in data]
        self.idx_to_classes = idx_to_classes
        self.classes_to_idx = classes_to_idx

    def __getitem__(self, idx):
        image_data = self.image_data[idx]
        ann_data = self.annotation_data[idx]

        image = Image.open(self.coco_path/'images'/image_data['file_name']).convert("RGB")
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

        # list all classes
        idx_to_classes = {cat['id']: cat['name'] for cat in annotations['categories']}
        classes_to_idx = {name: idx for (idx, name) in idx_to_classes.items()}

        # look for saved images
        saved_images = glob(str(coco_path/'images'/'*.jpg'))
        saved_images = set(img.split('/')[-1] for img in saved_images)

        # make the classes into a set for faster comparisons
        target_classes = set(classes_to_idx[cat] for cat in target_classes)

        # start a dict of data for images you already have
        annotation_data = {img['id']: [img] for img in annotations['images']
                           if img['file_name'] in saved_images}

        # add the annotation information to the data dict for each image
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            class_id = ann['category_id']
            if img_id in annotation_data and class_id in target_classes:
                annotation_data[img_id].append(ann)

        # don't include data if you only have the image information (no annotation data)
        annotation_data = {img_id: data for img_id, data in annotation_data.items() if len(data) > 1}

        return annotation_data, idx_to_classes, classes_to_idx

