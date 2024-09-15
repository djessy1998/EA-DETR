
from pathlib import Path
from torch.utils.data import Dataset
from .directory import HEDsecDirectory

import cv2
import torch
import numpy as np

class HardEventDSECDataset(Dataset):
    def __init__(self, root: Path, split="test"):

        self.root = root / ("train" if split in ['train', 'val'] else "test")

        self.sub_directories = [subdir for subdir in self.root.glob("*/") if subdir.is_dir()]

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        for f in self.sub_directories:
            directory = HEDsecDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = (0, directory.images.len - 1)

    def __getitem__(self, item):
        output = {}
        output['image'] = self.get_image(item)
        output['events'] = self.get_events(item)
        output['tracks'] = self.get_tracks(item)

        return output

    def __len__(self):
        return sum(self.directories[f.name].images.len for f in self.sub_directories)
    
    def get_image(self, index):
        index, directory = self.rel_index(index)
        image_files = directory.images.image_files
        image = cv2.imread(str(image_files[index]))
        image = torch.tensor(image).permute(2, 0, 1)
        return image

    def get_events(self, index):
        index, directory = self.rel_index(index)
        event_files = directory.events.event_files
        events = np.load(str(event_files[index]), allow_pickle=True)['events'][()]
        return events

    def get_tracks(self, index):
        index, directory = self.rel_index(index)
        tracks = directory.tracks.tracks
        tracks = tracks[tracks['index'] == index]
        return tracks        


    def rel_index(self, index):
        for f in self.sub_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if index >= img_idx_to_track_idx[0] and index <= img_idx_to_track_idx[1]:
                return index, self.directories[f.name]
            
        
    def compute_img_idx_to_track_idx(t_track, t_image):
        x, counts = np.unique(t_track, return_counts=True)
        i, j = (x.reshape((-1,1)) == t_image.reshape((1,-1))).nonzero()
        deltas = np.zeros_like(t_image)

        deltas[j] = counts[i]

        idx = np.concatenate([np.array([0]), deltas]).cumsum()
        return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")