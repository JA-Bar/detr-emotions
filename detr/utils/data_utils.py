import torch
from PIL import Image


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


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.cat([img.unsqueeze(0) for img in images])
    return images, labels


def labels_to_device(labels, device):
    for label in labels:
        for key in label:
            label[key] = label[key].to(device)
    return labels


def indices_to_device(indices, device):
    for i, _ in enumerate(indices):
        indices[i][0] = indices[i][0].to(device)
        indices[i][1] = indices[i][1].to(device)
    return indices

