import torch
import pytorch_lightning as pl
from util.box_ops import box_cxcywh_to_xyxy
import util.misc as utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

from prophesee_utils.metrics.coco_utils import coco_eval
from models import build_model
from util.detr_funcs import scale_for_map_computation
from util.visualize import visualize_pos_neg_boxes


class Detr(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.lr = args.lr

        self.index_im = 0
        self.total_spar = 0

        self.model, self.criterion, self.postprocessors = build_model(args)
        self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101_dc5', pretrained=True)
        
        num_classes = 9
        self.model.class_embed = torch.nn.Linear(in_features=self.model.class_embed.in_features, out_features=num_classes)

        self.param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

    def forward(self, image):
        features = self.model(image)
        return features
    
    def step(self, batch, batch_idx, mode):
        image_rgb, targets, image_events = batch[0], batch[1], batch[2]

        if(self.args.event):
            image = image_events
        else:
            image = image_rgb
            
        outputs = self(image)

        losses = None
        if mode != "test":
            loss_dict = self.criterion(outputs, targets)

            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            self.log(f'{mode}_loss_bbox', loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)
            self.log(f'{mode}_loss_clf', loss_dict_reduced['class_error'], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)
            self.log(f'lr', self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.b)

        # Postprocessing for mAP computation
        if mode != "train":
            results, targets = scale_for_map_computation(image_events, image_rgb, outputs, targets, self.postprocessors, self.index_im)

            #visualize_pos_neg_boxes(image, image_rgb, targets, results, results, self.index_im)

            getattr(self, f"{mode}_detections").extend([{k: v.cpu().detach() for k,v in r.items()} for r in results])
            getattr(self, f"{mode}_targets").extend([{k: v.cpu().detach() for k,v in t.items()} for t in targets])       

        self.index_im += 1
        
        # Work on 1 GPU only
        return losses

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

    def on_train_epoch_start(self):
        self.train_detections, self.train_targets = [], []

    def on_validation_epoch_start(self):
        self.val_detections, self.val_targets = [], []
     
    def on_test_epoch_start(self):
        self.test_detections, self.test_targets = [], []

    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")

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
    