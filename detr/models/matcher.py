"""
Basically Facebook's original implementation of the hungarian matcher, commented.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from detr.utils import box_ops


class HungarianMatcher(nn.Module):
    def __init__(self, lambda_classes, lambda_giou, lambda_l1):
        super().__init__()
        self.lambda_classes = lambda_classes
        self.lambda_giou = lambda_giou
        self.lambda_l1 = lambda_l1

    def forward(self, predictions, labels):
        batch_size, num_queries = predictions['logits'].shape[:-1]

        # TODO: wrap in no grad?

        # We flatten the batch dimension to make the computations easier
        # Due to images containing a different number of boxes, this way you
        # generalize
        flat_boxes_pred = predictions['bboxes'].flatten(0, 1)
        flat_boxes_labels = torch.cat([label['bboxes'] for label in labels])

        # GIoU loss is negative as a loss because you want to maximize GIoU
        pairwise_giou_loss = -box_ops.generalized_box_iou(flat_boxes_pred, flat_boxes_labels)
        pairwise_l1_loss = torch.cdist(flat_boxes_pred, flat_boxes_labels, p=1)
        boxes_loss = self.lambda_giou*pairwise_giou_loss + self.lambda_l1*pairwise_l1_loss

        # They use the prediction directly as a loss instead of cross-entropy
        flat_classes_pred = predictions['logits'].flatten(0, 1)
        flat_classes_labels = torch.cat([label['classes'] for label in labels])
        class_loss = -flat_classes_pred[:, flat_classes_labels]

        hungarian_loss = self.lambda_classes*class_loss + boxes_loss

        # Now to that you have the pairwise match loss, split it into the original batches
        # computation has to be on cpu because of scipy
        hungarian_loss = hungarian_loss.view(batch_size, num_queries, -1).cpu()

        boxes_per_batch = [label['bboxes'].size(0) for label in labels]

        # Iterate over every batch and solve for the assignment of indices that minimize loss
        # eg. (src2, src3, src1) that matches (tgt1, tgt3, tgt2)
        indices = [linear_sum_assignment(cost_matrix[n_batch]) for n_batch, cost_matrix in
                   enumerate(hungarian_loss.split(boxes_per_batch, -1))]

        indices = [(torch.as_tensor(pred_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64))
                   for pred_idx, tgt_idx in indices]

        return indices

