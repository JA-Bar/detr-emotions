import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

    train_transforms = transforms.get_train_transforms()
    val_transforms = transforms.get_val_transforms()

    logger.info("Loading the dataset...")
    if args.coco_path:
        train_dataset = CocoSubset(args.coco_path,
                                   args.target_classes,
                                   train_transforms,
                                   'train',
                                   args.val_split)
        val_dataset = CocoSubset(args.coco_path,
                                 args.target_classes,
                                 train_transforms,
                                 'val',
                                 args.val_split)
    else:
        raise ValueError("Specify path to coco or implement a custom dataset.")

    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              shuffle=True,
                              collate_fn=data_utils.collate_fn)

    val_loader = DataLoader(val_dataset,
                            args.batch_size,
                            shuffle=True,
                            collate_fn=data_utils.collate_fn)

    logger.info("Loading model...")
    model = DETR(args.num_classes, args.dim_model, args.n_heads, n_queries=args.n_queries)
    matcher = HungarianMatcher(args.lambda_matcher_classes,
                               args.lambda_matcher_giou,
                               args.lambda_matcher_l1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = AdamW(model.parameters(), args.lr)  # pending
    loss_fn = DETRLoss(args.lambda_loss_classes, args.lambda_loss_giou, args.lambda_loss_l1, args.num_classes)

    # writer = SummaryWriter(log_dir=Path(__file__)/'logs/tensorboard')
    # maybe image with boxes every now and then
    # maybe look into add_hparams

    # add checkpoint options and pretrain option
    # add tensorboard
    # add logging
    # add gradient accumulation

    logger.info("Starting training...")

    starting_epoch = 0
    for epoch in range(starting_epoch, args.epochs):
        # keep track of time
        # implement progress bar
        for images, labels in tqdm(train_loader, f'Epoch [{epoch}/{args.epochs}]'):

            images.to(device)
            # see labels moving to gpu

            output = model(images)
            matching_indices = matcher(output, labels)

            loss = loss_fn(output, labels, matching_indices)

            optim.zero_grad()
            loss.backward()
            optim.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('detr_train')

    # parser.add_argument('--target_classes', required=True)
    parser.add_argument('--target_classes', default="cat, dog")

    parser.add_argument('--num_classes', type=int, default=91)
    parser.add_argument('--coco_path', default=Path(__file__).parent/'../data/coco/')
    parser.add_argument('--val_split', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=32)

    parser.add_argument('--dim_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_queries', type=int, default=100)

    parser.add_argument('--lambda_matcher_classes', type=int, default=1)
    parser.add_argument('--lambda_matcher_giou', type=int, default=1)
    parser.add_argument('--lambda_matcher_l1', type=int, default=1)

    parser.add_argument('--lambda_loss_classes', type=int, default=1)
    parser.add_argument('--lambda_loss_giou', type=int, default=1)
    parser.add_argument('--lambda_loss_l1', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    args.target_classes = args.target_classes.replace(', ', ',').split(',')

    train(args)

print('hello world')
