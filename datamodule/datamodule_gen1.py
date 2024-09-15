import pytorch_lightning as L
import torch

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from datasets.gen1.gen1_od_dataset import GEN1DetectionDataset
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list

def collate_fn(batch):

    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)

    images = nested_tensor_from_tensor_list(samples)

    targets = [item[1] for item in batch]
    
    return [None , targets, images]

class GEN1DataModule(L.LightningDataModule):
    def __init__(self, args, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.args = args
    
    def setup(self, stage: str) -> None:

        if(stage == "fit" or stage is None):

            self.train = GEN1DetectionDataset(args=self.args, mode="train")
            self.val = GEN1DetectionDataset(args=self.args, mode="val")

        if(stage == "test" or stage is None):
            
            self.test = GEN1DetectionDataset(args=self.args, mode="test")
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)
    
