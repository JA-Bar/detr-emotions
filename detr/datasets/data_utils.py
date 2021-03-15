import torch
from albumentations.augmentations import bbox_utils
from PIL import Image


def convert_bboxes_format(bboxes, src_format, tgt_format, image_h, image_w):
    image_params = {'rows': image_h, 'cols': image_w, 'check_validity': True}
    bboxes = bbox_utils.convert_bboxes_to_albumentations(bboxes, src_format, **image_params)
    bboxes = bbox_utils.convert_bboxes_from_albumentations(bboxes, tgt_format, **image_params)
    return bboxes


def denormalize_tensor_image(image, mean='imagenet', std='imagenet', pillow_output=True):
    if mean == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406])
    else:
        mean = torch.tensor(mean)

    if std == 'imagenet':
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        std = torch.tensor(mean)

    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    image = (image*std) + mean
    image = image * 255
    image = image.permute(1, 2, 0)

    if pillow_output:
        image = Image.fromarray(image.numpy().astype('uint8'))

    return image

