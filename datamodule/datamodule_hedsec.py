import pytorch_lightning as L
import torch
import numpy as np
import cv2

from torch.utils.data import DataLoader
from datasets.hedsec_det.hard_event_dataset import HardEventDSECDataset
from datasets.hedsec_det.dataset_wrapper_he import HardEventDSECDetWrapper
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from util.misc import nested_tensor_from_tensor_list

def create_3_channel_tensor_from_events(event_dict, image_size):
    # Assume image_size is (H, W)
    H, W = image_size
    
    # Initialize an empty accumulator with the shape [H, W].
    event_accumulator = np.zeros((H, W), dtype=np.int32)

    # Apply slicing if the number of events is greater than 100000
    x = event_dict['x']
    y = event_dict['y']
    
    # Use numpy advanced indexing to accumulate events
    np.add.at(event_accumulator, (y, x), 1)
    
    # Directly stack the single channel 3 times to create a 3 channel image
    tensor_image_3ch = np.stack([event_accumulator]*3, axis=-1)
    
    # Convert to a PyTorch tensor
    tensor_image_3ch = torch.tensor(tensor_image_3ch, dtype=float)

    return tensor_image_3ch

def collate_fn(batch):
    samples = [item['image'] for item in batch]

    images = nested_tensor_from_tensor_list(samples)

    samples_histos = [create_3_channel_tensor_from_events(item['events'], image_size=(480, 640)) for item in batch]

    for i in range(len(samples_histos)):
        samples_histos[i] = cv2.normalize(samples_histos[i].numpy(), None, 0, 1, norm_type=cv2.NORM_MINMAX).astype(np.float32)     
        samples_histos[i] = torch.from_numpy(samples_histos[i])
        samples_histos[i] = samples_histos[i].permute(2, 0, 1)

    histos = torch.stack(samples_histos, 0)
    histos = nested_tensor_from_tensor_list(histos)

    tracks = [item['target'] for item in batch]

    return [images, tracks, histos]

class HEDsecDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()

        self.root_hedsec = root
        self.batch_size = batch_size
    
    def setup(self, stage: str) -> None:
        
        if(stage == "fit" or stage is None):
            assert NotImplementedError()

        if(stage == "test" or stage is None):   
            self.test = HardEventDSECDataset(self.root_hedsec, split="test")
            self.test = HardEventDSECDetWrapper(dsec_det_dataset=self.test, mode="test")
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0, collate_fn=collate_fn)