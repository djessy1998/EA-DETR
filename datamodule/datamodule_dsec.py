import pytorch_lightning as L
import torch
import numpy as np

import cv2
import os
import shutil

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import random_split, DataLoader
from datasets.dsec_det.dataset import DSECDet
from datasets.dsec_det.dataset_wrapper import DSECDetWrapper
from concurrent.futures import ProcessPoolExecutor
from util.misc import nested_tensor_from_tensor_list
from util.detr_funcs import create_3_channel_tensor_from_events


def collate_fn(batch):
    samples = [item['image'] for item in batch]
    images = nested_tensor_from_tensor_list(samples)

    samples_histos = [item['histo'] for item in batch]
    histos = nested_tensor_from_tensor_list(samples_histos)

    tracks = [item['target'] for item in batch]
    
    return [images, tracks, histos]

class DSECDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()

        self.root_dsec = root
        self.batch_size = batch_size
    
    def setup(self, stage: str) -> None:
        
        if(stage == "fit" or stage is None):

            train_set = DSECDet(self.root_dsec, split="train", sync="back", debug=True)

            train_set_size = int(len(train_set) * 0.8)
            val_set_size = len(train_set) - train_set_size

            self.train, self.val = random_split(train_set, [train_set_size, val_set_size])

            self.train = DSECDetWrapper(dsec_det_dataset=self.train, mode="train")
            self.val = DSECDetWrapper(dsec_det_dataset=self.val, mode="val")


        if(stage == "test" or stage is None):
            
            self.test = DSECDet(self.root_dsec, split="test", sync="back")
            self.test = DSECDetWrapper(dsec_det_dataset=self.test, mode="test")
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)
    
