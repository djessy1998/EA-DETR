import torch
import numpy as np
import cv2
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from util.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from util.detr_funcs import create_3_channel_tensor_from_events

class DSECDetWrapper(Dataset):
    def __init__(self, dsec_det_dataset, mode="train"):
        self.dataset = dsec_det_dataset

        self.mode = mode

        self.transform_train = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ToPureTensor()
        ])

        self.transform_val = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ToPureTensor()
        ])

    def __len__(self):
        return len(self.dataset)
    
    def normalize_bounding_boxes(self, target, w, h):
        box = box_xyxy_to_cxcywh(target)

        bounding_box_normalized = box

        bounding_box_normalized[[0, 2]] /= w
        bounding_box_normalized[[1, 3]] /= h

        return bounding_box_normalized

    def __getitem__(self, index):

        output = self.dataset[index]

        boxes = [[x, y, w, h] for _, x, y, w, h, _, _, _ in output['tracks']]
        labels = [int(class_id) for _, _, _, _, _, class_id, _, _ in output['tracks']]

        boxes_tv = tv_tensors.BoundingBoxes(boxes, format="XYWH", canvas_size=(480, 640))

        image = torch.tensor(output['image']).permute(2, 0, 1)

        histo = create_3_channel_tensor_from_events(output['events'], image_size=(480, 640))
        histo = cv2.normalize(histo.numpy(), None, 0, 1, norm_type=cv2.NORM_MINMAX).astype(np.float32)     
        histo = torch.from_numpy(histo)

        output['histo'] = histo.permute(2, 0, 1)
        
        if(len(boxes) == 0):
            if(self.mode == "train"):
                image, _ = self.transform_train(image, None)
            else:
                image, _ = self.transform_val(image, None)
        else:
            if(self.mode == "train"):
                image, boxes = self.transform_train(image, boxes_tv)
            else:
                image, boxes = self.transform_val(image, boxes_tv)

        output['image'] = image

        h, w = image.shape[1], image.shape[2]

        if len(output['tracks']) > 0:
            normalized_boxes = []
            for box in boxes:
                box_xyxy = box_xywh_to_xyxy(box)
                normalized_boxes.append(self.normalize_bounding_boxes(box_xyxy, w, h))

            boxes = torch.stack(normalized_boxes, dim=0)

            output['target'] = {'boxes': boxes, 'labels': torch.tensor(labels), 'orig_size': torch.as_tensor([int(480), int(640)])}
        
        else:

            output['target'] = {'boxes': torch.tensor([[0., 0., 0., 0.]]), 'labels': torch.tensor([0]), 'orig_size': torch.as_tensor([int(480), int(640)])}

        return output
