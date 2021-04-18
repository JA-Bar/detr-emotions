import torch
import torch.nn.functional as F
from torch import nn

from detr.utils import box_ops


class DETRLoss(nn.Module):
    def __init__(self, lambda_classes, lambda_giou, lambda_l1, num_classes):
        super().__init__()
        self.lambda_classes = lambda_classes
        self.lambda_giou = lambda_giou
        self.lambda_l1 = lambda_l1
        self.no_class_index = num_classes

    def forward(self, predictions, labels, indices):
        """Compute the loss.

        Args:
            predictions: Dictionary of predictions of the model.
                         The two keys needed are "logits" and "bboxes".
                Logits:
                    Class predictions of the model. Tensor of shape [batch_size, n_object_queries, n_classes].
                Bboxes:
                    Boxes predictions of the model. Tensor of shape [batch_size, n_object_queries, 4].

            labels: Dictionary of labels corresponsing to the predictions.
                    The two keys needed are "classes" and "bboxes".
                Classes:
                    Class labels of the model. List of size [batch_size] that contains Tensors of shape [n_objects_in_image]
                Bboxes:
                    Boxes labels of the model. Tensor of shape [batch_size, n_object_queries, 4].

            indices: List of tuples corresponsing to the pairing indices.
        """
        class_loss = self.classification_loss(predictions, labels, indices)
        giou_loss, l1_loss = self.bbox_losses(predictions, labels, indices)

        # compute the box loss
        total_loss = class_loss*self.lambda_classes + giou_loss*self.lambda_giou + l1_loss*self.lambda_l1

        return total_loss

    def classification_loss(self, predictions, labels, indices):
        # extract the indices of both: predictions and labels into separate Tensors
        # pred/labels_idx hold the indices of each corresponsing matched pair
        # pred/labels_batch_idx is just a Tensor with the batch that each index belongs to
        pred_all_indices = [(torch.full_like(pred, batch), pred) for batch, (pred, _) in enumerate(indices)]
        pred_batch_idx, pred_idx = map(torch.cat, zip(*pred_all_indices))

        # compute the classification loss: -log(P(Ci))
        classes_pred = predictions['logits']
        classes_labels = labels['classes']

        target_classes = torch.cat([batch[J] for batch, (_, J) in zip(classes_labels, indices)])
        all_target_classes = torch.full(
            (classes_pred.size(0), classes_pred.size(1)), self.no_class_index, dtype=torch.int64
        )
        all_target_classes[pred_batch_idx, pred_idx] = target_classes

        class_loss = F.cross_entropy(classes_pred.transpose(1, 2), all_target_classes)
        return class_loss

    @staticmethod
    def bbox_losses(predictions, labels, indices):
        """Returns (giou_loss, l1_loss) of predictions and labels bounding boxes."""

        pred_all_indices = [(torch.full_like(pred, batch), pred) for batch, (pred, _) in enumerate(indices)]
        pred_batch_idx, pred_idx = map(torch.cat, zip(*pred_all_indices))

        boxes_pred = predictions['bboxes']  # [batch_size, n_object_queries, 4]
        boxes_labels = labels['bboxes']  # [batch_size, n_objects_in_image, 4]
        boxes_pred = boxes_pred[pred_batch_idx, pred_idx]

        # compute the generalized IoU loss
        flat_boxes_pred = boxes_pred.flatten(0, 1)
        flat_boxes_pred = box_ops.box_cxcywh_to_xyxy(flat_boxes_pred)

        flat_boxes_labels = torch.cat(boxes_labels)
        flat_boxes_labels = box_ops.box_cxcywh_to_xyxy(flat_boxes_labels)

        giou_loss = box_ops.generalized_box_iou(flat_boxes_pred, flat_boxes_labels)
        giou_loss = torch.diag(giou_loss).sum()

        # compute the L1 loss
        l1_loss = F.l1_loss(flat_boxes_pred, flat_boxes_labels)

        return giou_loss, l1_loss

