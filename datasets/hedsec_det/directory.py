import numpy as np
from functools import lru_cache

def extract_number(file_path):
    filename = file_path.stem
    numeric_part = filename.split('_')[-1]
    return int(numeric_part)

class BaseDirectory:
    def __init__(self, root):
        self.root = root


class HEDsecDirectory:
    def __init__(self, root):
        self.root = root
        self.images = ImageDirectory(root / "rgb")
        self.events = EventDirectory(root / "events")
        self.tracks = TracksDirectory(root / "ground_truth")


class ImageDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def image_files(self):
        file_paths = list((self.root).glob("*.png"))
        sorted_file = sorted(file_paths, key=extract_number)
        return sorted_file
    
    @property
    @lru_cache(maxsize=1)
    def len(self):
        return len(list((self.root).glob("*.png")))


class EventDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def event_files(self):
        file_paths = list((self.root).glob("*.npz"))
        sorted_file = sorted(file_paths, key=extract_number)
        return sorted_file


class TracksDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def tracks(self):
        return np.load(self.root / "tracks.npy")