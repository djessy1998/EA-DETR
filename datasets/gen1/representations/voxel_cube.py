import torch
import numpy as np

from numpy.lib.recfunctions import structured_to_unstructured
from representations.repr import Representation

class VoxelCube(Representation):
    def __init__(self, args) -> None:
        super().__init__()

        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size
        self.quantization_size = [args.sample_size // args.T,1,1]
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

    def get_str_repr(self):
        return "voxelcube"

    def get_item(self, index):

        (coords, feats), target = self.samples[index]
        
        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.quantized_h, self.quantized_w, self.C)
            )
        sample = sample.coalesce().to_dense().permute(0,3,1,2)

        return sample, target

    def create_data(self, events):

        events['t'] -= events['t'][0]

        feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)
        # 
        coords = torch.from_numpy(structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

        # Bin the events on T timesteps
        coords = torch.floor(coords/torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h-1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w-1)

        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        tbin_feats = ((events['p']+1) * (tbin_coords+1)) - 1

        feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*self.tbin).to(bool)

        return coords.to(torch.int16), feats.to(bool)