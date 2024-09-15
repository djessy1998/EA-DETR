import copy
import functools
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from torch import nn
from torch.profiler import ProfilerActivity, profile

import util.misc as utils
from models import build_model
from models.detr import SetCriterion
from models.matcher import HungarianMatcher
from models_lightning.detr_lightning import Detr
from prophesee_utils.metrics.coco_utils import coco_eval
from util.box_ops import box_cxcywh_to_xyxy
from util.detr_funcs import *
from util.misc import (create_pos_neg_teacher, filter_small_objects,
                       filter_with_events, get_negative_idx, get_small_bboxes,
                       nested_tensor_from_tensor_list)
from util.visualize import visualize_pos_neg_boxes


class DetrDistill(L.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.lr = args.lr

        self.index_im = 0

        self.small_objects = 0
        self.medium_objects = 0
        self.large_objects = 0
        
        self.teacher_model = Detr.load_from_checkpoint(args=args, checkpoint_path=args.teacher_detr_root)

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_backbone = self.teacher_model.model.backbone

        self.student_model, self.criterion, self.postprocessors = build_model(args)

        self.student_model = Detr.load_from_checkpoint(args=args, checkpoint_path=args.teacher_detr_root)
        self.student_backbone = self.student_model.model.backbone

        self.backbone_loss = nn.MSELoss()
        self.hung_match = HungarianMatcher()

        self.factor_distill_backbone = 1
        self.factor_distill_logits = 1

        self.param_dicts = [
            {"params": [p for n, p in self.student_model.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.student_model.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]


    def forward(self, x):
        return self.student_model(x)
    
    def step(self, batch, batch_idx, mode):

        image_rgb, targets, image_events = batch[0], batch[1], batch[2]

        # Student's features
        features_pred, pos_events = self.student_backbone(image_events)
        src_events, mask_events = features_pred[-1].decompose()

        # Student's transformer
        hs_events = self.student_model.model.transformer(self.student_model.model.input_proj(src_events), mask_events, self.student_model.model.query_embed.weight, pos_events[-1])[0]

        weight_dict = self.criterion.weight_dict

        # Student's output
        outputs_class_s = self.student_model.model.class_embed(hs_events)
        outputs_coord_s = self.student_model.model.bbox_embed(hs_events).sigmoid()
        outputs_s = {'pred_logits': outputs_class_s[-1], 'pred_boxes': outputs_coord_s[-1]}

        backbone_loss = 0
        total_loss_logits = 0

        if(mode == "train"):

            # Teacher's features
            features_target, pos_rgb = self.teacher_backbone(image_rgb)
            src_rgb, mask_rgb = features_target[-1].decompose()
    
            if(self.args.features):

                mask = (image_events.tensors != 0).float()
                
                mask = nn.functional.interpolate(mask, size=src_rgb.shape[2:], mode='nearest')

                # Masked MSE Loss to distil teacher's features
                backbone_loss = self.backbone_loss(src_events, src_rgb)

                backbone_loss = (backbone_loss * mask).mean()

                self.log(f'{mode}_bcbloss', backbone_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)

            # Teacher's transformer
            hs_rgb = self.teacher_model.model.transformer(self.teacher_model.model.input_proj(src_rgb), mask_rgb, self.teacher_model.model.query_embed.weight, pos_rgb[-1])[0]

            # Teacher's output
            outputs_class_t = self.teacher_model.model.class_embed(hs_rgb)
            outputs_coord_t = self.teacher_model.model.bbox_embed(hs_rgb).sigmoid()
            outputs_t = {'pred_logits': outputs_class_t[-1], 'pred_boxes': outputs_coord_t[-1]}

            if(self.args.distill_logits):

                #Application de Hungarian pour récupérer les boxes positives du teacher
                ind_t = self.hung_match(outputs_t, targets)

                #Boxes négatives du teacher
                ind_neg_t = get_negative_idx([t[0] for t in ind_t])

                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results_t = self.postprocessors['bbox'](outputs_t, orig_target_sizes)

                ind_neg_t = filter_with_events(results_t, ind_neg_t, image_events)

                results_pos_t, results_neg_t = create_pos_neg_teacher(outputs_t, results_t, ind_t, ind_neg_t)
  
                # Loss KD pos
                loss_pos = self.criterion(outputs_s, results_pos_t)
                loss_dict_pos_reduced = utils.reduce_dict(loss_pos)
                loss_dict_pos_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_pos_reduced.items() if k in weight_dict}
                losses_pos_reduced_scaled = sum(loss_dict_pos_reduced_scaled.values())
                lmatchpos = losses_pos_reduced_scaled.item()

                self.log(f'{mode}_loss_pos', lmatchpos, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)
                
                # Loss KD neg
                loss_neg = self.criterion(outputs_s, results_neg_t)
                loss_dict_neg_reduced = utils.reduce_dict(loss_neg)
                loss_dict_neg_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_neg_reduced.items() if k in weight_dict}
                losses_neg_reduced_scaled = sum(loss_dict_neg_reduced_scaled.values())
                lmatchneg = losses_neg_reduced_scaled.item()

                self.log(f'{mode}_loss_neg', lmatchneg, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)

                total_loss_logits = losses_pos_reduced_scaled + losses_neg_reduced_scaled
            
        # Loss detection DETR
        loss_dict = self.criterion(outputs_s, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        self.log(f'{mode}_loss_bbox', loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)
        self.log(f'{mode}_loss_clf', loss_dict['class_error'], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)

        # Postprocessing for mAP computation
        if mode != "train":

            results, targets = scale_for_map_computation(image_events, image_rgb, outputs_s, targets, self.postprocessors, self.index_im)

            getattr(self, f"{mode}_detections").extend([{k: v.cpu().detach() for k,v in r.items()} for r in results])
            getattr(self, f"{mode}_targets").extend([{k: v.cpu().detach() for k,v in t.items()} for t in targets])

        total_loss = loss_value
        if(self.args.features):
            total_loss += self.factor_distill_backbone * backbone_loss
        if(self.args.distill_logits):
            total_loss += self.factor_distill_logits * total_loss_logits

        self.index_im += 1

        return total_loss     

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")

    def on_mode_epoch_end(self, mode):
        print()
        if mode != "train":
            print(f"[{self.current_epoch}] {mode} results:")
            
            targets = getattr(self, f"{mode}_targets")
            detections = getattr(self, f"{mode}_detections")

            if detections == []:
                print("No detections")
                return

            h, w = 480, 640
            stats = coco_eval(
                targets, 
                detections, 
                height=h, width=w, 
                labelmap=('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'background'))

            keys = [
                'val_AP_IoU_0_5_to_0_95', 'val_AP_IoU=.5', 'val_AP_IoU=.75', 
                'val_AP_small', 'val_AP_medium', 'val_AP_large',
                'val_AR_det=1', 'val_AR_det=10', 'val_AR_det=100',
                'val_AR_small', 'val_AR_medium', 'val_AR_large',
            ]
            stats_dict = {k:v for k,v in zip(keys, stats)}

            self.log_dict(stats_dict, batch_size=self.args.b)

    def on_validation_epoch_start(self):
        self.val_detections, self.val_targets = [], []

    def on_test_epoch_start(self):
        self.test_detections, self.test_targets = [], []

    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")

    def configure_optimizers(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Number of parameters:', n_parameters)
        
        optimizer = torch.optim.AdamW(
            self.param_dicts, 
            lr=self.lr,
            weight_decay=self.args.wd,
        )
    
        return [optimizer]