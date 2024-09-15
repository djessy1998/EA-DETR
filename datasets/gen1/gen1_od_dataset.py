import os
import tqdm
import torch
import numpy as np
import cv2

from torchvision.transforms import v2
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader
from .representations.histogram import Histogram
from torchvision import tv_tensors
from .transforms import normalize_image_0_1, normalize_bounding_boxes


class GEN1DetectionDataset(Dataset):
    def __init__(self, args, mode="train"):
        
        self.mode = mode
        self.sample_size = args.sample_size
        self.h, self.w = args.image_shape
        self.args = args

        self.repr = Histogram(args=args)

        self.image_size = (240, 304)

        self.transforms = v2.Compose([
            v2.Resize(size=(480, 608)),
            v2.ToPureTensor()
        ])

        for size in args.acc_augmentation:
            save_file_name = f"gen1_{mode}_{size//1000}_{self.repr.get_str_repr()}.pt"
            save_file = os.path.join(args.root_gen1, save_file_name)
        
            if os.path.isfile(save_file):
                self.repr.set_samples(size, torch.load(save_file))
                print("File loaded.")
            else:
                data_dir = os.path.join(args.root_gen1, mode)
                self.repr.set_samples(size, self.build_dataset(data_dir, save_file))
                torch.save(self.repr.samples_dict[size], save_file)
                print(f"Done! File saved as {save_file}")

            
    def __getitem__(self, index):
        sample, target = self.repr.get_item(index)

        target['orig_size'] = torch.as_tensor([480, 608])

        boxes = tv_tensors.BoundingBoxes(target['boxes'], format="XYXY", canvas_size=(240, 304))

        sample, boxes = self.transforms(sample, boxes)

        target['boxes'] = boxes

        '''
        image = sample.detach().cpu().permute(1, 2, 0).numpy()
        image = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        for i in range(len(target['boxes'])):
            bbx_target = boxes
            cv2.rectangle(image, (int(bbx_target[i][0]), int(bbx_target[i][1])), (int(bbx_target[i][2]), int(bbx_target[i][3])), [0,0,255], 2, cv2.LINE_AA)

        cv2.imshow("title", image)
        cv2.waitKey(0)
        '''

        for idx, histo in enumerate(sample):
            sample[idx] = torch.from_numpy(cv2.normalize(sample[idx].numpy(), None, 0, 1, norm_type=cv2.NORM_MINMAX).astype(np.float32)) 
        
        target_augmented = normalize_bounding_boxes(target, sample.shape[2], sample.shape[1])


        return sample, target_augmented


    
    def __len__(self):
        return len(next(iter(self.repr.samples_dict.values())))

        
    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                        if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        for file_name in files:
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])
            
            samples.extend([sample for b in boxes_per_ts if (sample := self.create_sample(video,b)) is not None])
            pbar.update(1)

        pbar.close()
        torch.save(samples, save_file)
        print(f"Done! File saved as {save_file}")
        return samples
        
    def create_sample(self, video, boxes):
        ts = boxes['t'][0]
        video.seek_time(ts-self.sample_size)
        events = video.load_delta_t(self.sample_size)
        
        targets = self.create_targets(boxes)
        
        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.repr.create_data(events), targets)
        
    def create_targets(self, boxes):
        torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32).copy())
        
        # keep only last instance of every object per target
        _,unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True) # keep last unique objects
        unique_indices = np.flip(-(unique_indices+1))
        torch_boxes = torch_boxes[[*unique_indices]]
        
        torch_boxes[:, 2:] += torch_boxes[:, :2] # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)
        
        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:,2]-torch_boxes[:,0] != 0) & (torch_boxes[:,3]-torch_boxes[:,1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]
        
        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {'boxes': torch_boxes, 'labels': torch_labels}