import albumentations as A
import albumentations.pytorch.transforms


def get_train_transforms():
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.SmallestMaxSize(800),
        A.RandomResizedCrop(800, 1333, scale=(0.8, 1.0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    return transforms


def get_val_transforms():
    transforms = A.Compose([
        A.PadIfNeeded(800, 1333),
        A.CenterCrop(800, 1333),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    return transforms

