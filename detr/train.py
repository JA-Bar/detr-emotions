import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
from tqdm import tqdm

import detr.logs.logger as log

from detr.datasets import transforms
from detr.datasets.coco_subset import CocoSubset
from detr.models.detr import DETR
from detr.models.losses import DETRLoss
from detr.models.matcher import HungarianMatcher
from detr.utils import data_utils


def train(args):
    logger = log.get_logger(__name__)

    with open(Path(args.config_base_path, args.config).with_suffix(".yaml"), 'r') as f:
        config = yaml.safe_load(f)

    train_transforms = transforms.get_train_transforms()
    val_transforms = transforms.get_val_transforms()

    logger.info("Loading the dataset...")
    if config['dataset']['name'] == 'coco_subset':
        train_dataset = CocoSubset(config['dataset']['coco_path'],
                                   config['dataset']['target_classes'],
                                   train_transforms,
                                   'train',
                                   config['dataset']['train_val_split'])

        val_dataset = CocoSubset(config['dataset']['coco_path'],
                                 config['dataset']['target_classes'],
                                 val_transforms,
                                 'val',
                                 config['dataset']['train_val_split'])
    else:
        raise ValueError("Dataset name not recognized or implemented")

    train_loader = DataLoader(train_dataset,
                              config['training']['batch_size'],
                              shuffle=True,
                              collate_fn=data_utils.collate_fn)

    val_loader = DataLoader(val_dataset,
                            config['training']['batch_size'],
                            shuffle=True,
                            collate_fn=data_utils.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model...")
    model = DETR(config['dataset']['num_classes'],
                 config['model']['dim_model'],
                 config['model']['n_heads'],
                 n_queries=config['model']['n_queries'])

    if args.mode == 'pretrained':
        model.load_demo_state_dict('data/state_dicts/detr_demo.pth')

    model.to(device)

    matcher = HungarianMatcher(config['losses']['lambda_matcher_classes'],
                               config['losses']['lambda_matcher_giou'],
                               config['losses']['lambda_matcher_l1'])

    optim = AdamW(model.parameters(), config['training']['lr'])  # pending
    loss_fn = DETRLoss(config['losses']['lambda_loss_classes'],
                       config['losses']['lambda_loss_giou'],
                       config['losses']['lambda_loss_l1'],
                       config['dataset']['num_classes'])

    # writer = SummaryWriter(log_dir=Path(__file__)/'logs/tensorboard')
    # maybe image with boxes every now and then
    # maybe look into add_hparams

    # add checkpoint options and pretrain option
    # add tensorboard
    # add logging
    # add gradient accumulation

    logger.info("Starting training...")

    starting_epoch = 0
    for epoch in range(starting_epoch, config['training']['epochs']):
        for images, labels in tqdm(train_loader, f"Epoch [{epoch}/{config['training']['epochs']}]"):

            images = images.to(device)
            labels = data_utils.labels_to_device(labels, device)
            # see labels moving to gpu

            output = model(images)
            matching_indices = matcher(output, labels)
            matching_indices = data_utils.indices_to_device(matching_indices, device)

            loss = loss_fn(output, labels, matching_indices)

            optim.zero_grad()
            loss.backward()
            optim.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('detr_train')

    # parser.add_argument('--target_classes', required=True)
    # parser.add_argument('--target_classes', default="cat, dog")

    parser.add_argument('--mode', default='pretrined', choices=['pretrained', 'checkpoint', 'from_scratch'])
    parser.add_argument('--config_base_path', default='configs/')
    parser.add_argument('--config', default='coco_fine_tune')

    # parser.add_argument('--num_classes', type=int, default=91)
    # parser.add_argument('--coco_path', default=Path(__file__).parent/'../data/coco/')
    # parser.add_argument('--val_split', type=float, default=0.1)

    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--effective_batch_size', type=int, default=32)

    # parser.add_argument('--dim_model', type=int, default=256)
    # parser.add_argument('--n_heads', type=int, default=8)
    # parser.add_argument('--n_queries', type=int, default=100)

    # parser.add_argument('--lambda_matcher_classes', type=int, default=1)
    # parser.add_argument('--lambda_matcher_giou', type=int, default=1)
    # parser.add_argument('--lambda_matcher_l1', type=int, default=1)

    # parser.add_argument('--lambda_loss_classes', type=int, default=1)
    # parser.add_argument('--lambda_loss_giou', type=int, default=1)
    # parser.add_argument('--lambda_loss_l1', type=int, default=1)

    # parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    # args.target_classes = args.target_classes.replace(', ', ',').split(',')

    train(args)

print('hello world')
