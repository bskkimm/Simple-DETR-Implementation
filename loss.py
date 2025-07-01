import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from config import dataset_config
from config import model_config
from config import train_config

class DETRLoss(nn.Module):
    def __init__(self,
                 num_classes=dataset_config['num_classes'],
                 decoder_layers=model_config['decoder_layers'],
                 num_queries=model_config['num_queries'],
                 cls_cost_weight=model_config['cls_cost_weight'],
                 l1_cost_weight=model_config['l1_cost_weight'],
                 giou_cost_weight=model_config['giou_cost_weight'],
                 bg_class_idx=dataset_config['bg_class_idx'],
                 bg_class_weight=model_config['bg_class_weight'],
                 nms_threshold=model_config['nms_threshold']):
        super().__init__()
        self.num_classes = num_classes
        self.num_decoder_layers = decoder_layers
        self.num_queries = num_queries
        self.cls_cost_weight = cls_cost_weight
        self.l1_cost_weight = l1_cost_weight
        self.giou_cost_weight = giou_cost_weight
        self.bg_class_idx = bg_class_idx
        self.bg_class_weight = bg_class_weight
        self.nms_threshold = nms_threshold

    def compute_hungarian_matching(self, pred_logits, pred_boxes, targets):
        # pred_logits = [B, num_queries, num_classes]

        batch_size = pred_logits.shape[0]
        num_queries = pred_logits.shape[1]

        class_prob = pred_logits.reshape(-1, self.num_classes).softmax(dim=-1)
        pred_boxes = pred_boxes.reshape(-1, 4)
        # class_prob, pred_boxes = (B*num_queries, num_classes), (B*num_queries, 4)

        target_labels = torch.cat([t['labels'] for t in targets])
        target_boxes = torch.cat([t['boxes'] for t in targets])
        # e.g., two objs [1, 11] and an obj [5] -> target_labels = torch.tensor([1,11,5])
        # target_boxes = torch.tensor([torch.size(4),torch.size(4), torch.size(4)]       

        ## Classification cost
        cost_classification = -class_prob[:, target_labels]
        # cost_classifi = (B*num_queries, total_num_objs_in_batch). if case above -> (B*num_queries,3)

        # To calculate the GIoU, need to transform  cx,cy,w,h into 'x,y,x,y'
        pred_boxes_xyxy = torchvision.ops.box_convert(pred_boxes, 'cxcywh', 'xyxy')
        

        ## Bbox cost (L1 + GIoU) 
        # L1 cost
        cost_l1 = torch.cdist(pred_boxes_xyxy, target_boxes, p=1)
        # cost_l1 = (B*num_queries, total_num_objs_in_batch)

        # GIoU cost
        cost_giou = -torchvision.ops.generalized_box_iou(pred_boxes_xyxy, target_boxes)
        # cost_giou = (B*num_queries, total_num_objs_in_batch)

        ### Gross loss = classification cost + L1 cost + GIoU cost
        total_cost = (self.cls_cost_weight * cost_classification +
                      self.l1_cost_weight * cost_l1 +
                      self.giou_cost_weight * cost_giou)

        total_cost = total_cost.reshape(batch_size, self.num_queries, -1).cpu()
        # total_cost = (B, num_queries, total_num_objs_in_batch)

        num_targets_per_image = [len(t['labels']) for t in targets]
        # e.g., num_targets_per_image = [2, 1]
        total_cost_per_image = total_cost.split(num_targets_per_image, dim=-1)
        # e.g., total_cost_per_image[0] = (B, num_queries, 2) <- 2 GTs from image 0
        # e.g., total_cost_per_image[1] = (B, num_queries, 1) <- 1 GT from image 1

        match_indices = []
        # This function gives the best matchings between predicted queries and GTs, 
        # minimizing the total cost:
        # pred_inds: indices into the 25 queries
        # tgt_inds: indices into the N_b GTs
        for b in range(batch_size):
            pred_inds, tgt_inds = linear_sum_assignment(total_cost_per_image[b][b])
            # picking the b-th image among the batch of B=2, from the b-th cost tensor.
            match_indices.append((
                torch.as_tensor(pred_inds, dtype=torch.int64),
                torch.as_tensor(tgt_inds, dtype=torch.int64)
            ))
            # e.g., match_indices = [
            #       (tensor([ 5, 13]), tensor([0, 1])),  # image 0: query 5→GT 0, query 13→GT 1
            #       (tensor([2]), tensor([0]))]           # image 1: query 2→GT 0
        return match_indices

    def compute_losses(self, pred_logits, pred_boxes, targets, match_indices):
        batch_size = pred_logits.shape[0]
        losses = defaultdict(list)
        classification_losses = []  # <<< ADDED
        bbox_losses = []            # <<< ADDED
        bbox_giou_losses = []       # <<< ADDED

        for b in range(batch_size):
            pred_idx, tgt_idx = match_indices[b]
            # e.g., ([3, 7], [0, 1])

            # Create a tensor of shape (num_queries,) filled with the background class index.
            # Used to label all unassigned queries as background by default.
            target_classes = torch.full((self.num_queries,), self.bg_class_idx,
                                        dtype=torch.int64, device=pred_logits.device)
            # target_classes = [0]*num_queries

            # For matched queries, update the target class with the corresponding GT label.
            target_classes[pred_idx] = targets[b]['labels'][tgt_idx]
            # e.g., targets[b]['labels'] = tensor([5, 2])
            # target_classes = [0, 0, 0, 5, 0, 0, 0 ,2, ... 0]
            #              idx  0  1  2  3  4  5  6  7  ... 24

            # Define the class weights for cross-entropy.
            cls_weights = torch.ones(self.num_classes, device=pred_logits.device)
            cls_weights[self.bg_class_idx] = self.bg_class_weight
            #cls_weights = [0.1, 1, 1, ...,  1],
            #            idx  0   1  2  .... 20

            # (1) Classification loss --> return scalar
            #loss_cls = F.cross_entropy(pred_logits[b], target_classes, weight=cls_weights)
            # Don't reduce the loss here, will be done later.
            loss_cls = F.cross_entropy(pred_logits[b], target_classes, weight=cls_weights, reduction='none')  # <<< CHANGED
            classification_losses.append(loss_cls)  # <<< CHANGED

            matched_pred_boxes = pred_boxes[b][pred_idx]
            target_boxes = targets[b]['boxes'][tgt_idx]
            #matched_pred_boxes, target_boxes = [num_matched_queries, 4] 

            # Convert predicted boxes from cx,cy,w,h to x1,y1,x2,y2) for GIoU computation.
            pred_xyxy = torchvision.ops.box_convert(matched_pred_boxes, 'cxcywh', 'xyxy')

            # (2) Bboxes loss --> return scalar
            loss_bbox = F.l1_loss(pred_xyxy, target_boxes, reduction='none').sum(dim=1)  # <<< CHANGED
            loss_giou = torchvision.ops.generalized_box_iou_loss(pred_xyxy, target_boxes, reduction='none')  # <<< CHANGED
            #loss_bbox = F.l1_loss(pred_xyxy, target_boxes, reduction='none').sum() / matched_pred_boxes.shape[0]
            #loss_giou = torchvision.ops.generalized_box_iou_loss(pred_xyxy, target_boxes).sum() / matched_pred_boxes.shape[0]

            bbox_losses.append(loss_bbox)
            bbox_giou_losses.append(loss_giou)
            #bbox_loss = (loss_bbox.sum(dim=1) + loss_giou)  # <<< CHANGED
            #bbox_losses.append(bbox_loss)  # <<< CHANGED

            # Multiply each weight
            #losses['classification'].append(loss_cls * self.cls_cost_weight)
            #losses['bbox_regression'].append(loss_bbox * self.l1_cost_weight + loss_giou * self.giou_cost_weight)

        # Concatenate all per-image losses and average over batch
        all_cls_loss = torch.cat(classification_losses).mean()
        all_l1_loss = torch.cat(bbox_losses).mean()
        all_giou_loss = torch.cat(bbox_giou_losses).mean()

        los_cls = all_cls_loss * self.cls_cost_weight
        los_bbox = all_l1_loss * self.l1_cost_weight + all_giou_loss * self.giou_cost_weight
        
        return los_cls, los_bbox

    def forward(self, pred_classes, pred_bboxes, targets, training=True, score_thresh=0.0, use_nms=False):
        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if training:
            for decoder_idx in range(self.num_decoder_layers):
                cls_out = pred_classes[decoder_idx]
                box_out = pred_bboxes[decoder_idx]

                with torch.no_grad():
                    match_indices = self.compute_hungarian_matching(cls_out, box_out, targets)

                loss_cls, loss_bbox = self.compute_losses(cls_out, box_out, targets, match_indices)
                losses['classification'].append(loss_cls)
                losses['bbox_regression'].append(loss_bbox)
            # Average losses across all decoder layers


            detr_output['loss'] = losses

        else:
            # From the final decoder
            cls_out = pred_classes[-1] 
            box_out = pred_bboxes[-1]
            # (B, num_queries, num_classes), (B, num_queries, 4)
            prob = F.softmax(cls_out, -1)

            # Delete the background class
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels += 1
                # scores, labels = (B, num_queries)
            else:
                scores, labels = prob[..., :-1].max(-1)

            boxes = torchvision.ops.box_convert(box_out, 'cxcywh', 'xyxy')
            # boxes = (B, num_queries, 4)

            # Iterate through each image in the batch
            for b in range(boxes.shape[0]):
                score_b, label_b, box_b = scores[b], labels[b], boxes[b] # confidence, class, boxes
                # (num_queries,)
                keep = score_b >= score_thresh
                #e.g.,
                    # score_b = tensor([0.9, 0.1, 0.5, 0.7])
                    # score_thresh = 0.5
                    # keep = tensor([True, False, True, True])
                score_b, label_b, box_b = score_b[keep], label_b[keep], box_b[keep]
                # score_b  = tensor([0.9, 0.5, 0.7])  # kept

                if use_nms:
                    keep_nms = torchvision.ops.batched_nms(box_b, score_b, label_b, self.nms_threshold)
                    score_b, label_b, box_b = score_b[keep_nms], label_b[keep_nms], box_b[keep_nms]

                detections.append({"boxes": box_b, "scores": score_b, "labels": label_b})

            detr_output['detections'] = detections

        return detr_output
