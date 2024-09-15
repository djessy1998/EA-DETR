from pathlib import Path 
import cv2
import numpy as np
import torch

from .directory import DSECDirectory
from .preprocessing import compute_img_idx_to_track_idx
from .io import extract_from_h5_by_timewindow
from .visualize import render_object_detections_on_image, render_events_on_image
from .label import CLASSES


class DSECDet:
    def __init__(self, root: Path, split: str="train", sync: str="front", debug: bool=False, split_config=None):
        """
        root: Root to the the DSEC dataset (the one that contains 'train' and 'test'
        split: Can be one of ['train', 'test']
        window_size: Number of microseconds of data
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.

        Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for 
        sync='front' and sync='back', and these are visualized below.

        Legend: 
        . = events
        | = image
        L = label

        sync='front'
        -------> time
        .......|
               L

        sync='back'
        -------> time
        |.......
               L
        
        """
        assert root.exists()
        assert split in ['train', 'test', 'val']
        assert (root / split).exists()
        assert sync in ['front', 'back']

        self.debug = debug
        self.classes = CLASSES

        self.split = split

        self.root = root / ("train" if split in ['train', 'val'] else "test")
        self.sync = sync

        self.height = 480
        self.width = 640

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        if split_config is None:
            self.subsequence_directories = list(self.root.glob("*/"))
        else:
            self.subsequence_directories = [self.root / s for s in split_config[split]]

        self.subsequence_directories = sorted(self.subsequence_directories, key=self.first_time_from_subsequence)

        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = compute_img_idx_to_track_idx(directory.tracks.tracks['t'],
                                                                           directory.images.timestamps)
            
        
                   
    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]

    def __len__(self):
        return sum(len(v)-1 for v in self.img_idx_track_idxs.values())

    def __getitem__(self, item):
        output = {}
        output['image'] = self.get_image(item)
        output['events'] = self.get_events(item)
        output['tracks'] = self.get_tracks(item)

        return output

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            i_0 = index - 1
            i_1 = index
        else:
            assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = index + 1

        return i_0, i_1

    def get_tracks(self, index, mask=None, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        idx0, idx1 = img_idx_to_track_idx[i_1]
        tracks = directory.tracks.tracks[idx0:idx1]

        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]

        return tracks

    def get_events(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]

        events = extract_from_h5_by_timewindow(directory.events.event_file, t_0, t_1)
        return events

    def get_image(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        image_files = directory.images.image_files_distorted
        image = cv2.imread(str(image_files[index]))
        return image

    def rel_index(self, index, directory_name=None):
        if directory_name is not None:
            img_idx_to_track_idx = self.img_idx_track_idxs[directory_name]
            directory = self.directories[directory_name]
            return index, img_idx_to_track_idx, directory

        for f in self.subsequence_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if len(img_idx_to_track_idx)-1 <= index:
                index -= (len(img_idx_to_track_idx)-1)
                continue
            else:
                return index, img_idx_to_track_idx, self.directories[f.name]
        else:
            raise ValueError
        
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