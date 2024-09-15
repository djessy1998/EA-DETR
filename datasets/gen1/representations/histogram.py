import torch
import random

class Histogram():
    def __init__(self, args) -> None:
        self.T = args.T
        self.C = 2
        self.h, self.w = args.image_shape

        self.samples_dict = {key: [] for key in args.acc_augmentation}

    def get_str_repr(self):
        return "histo"
    
    def set_samples(self, key, samples):
        self.samples_dict[key] = samples

    def get_item(self, index):

        choice_acc = random.choice(list(self.samples_dict))

        (coords, feats), target = self.samples_dict[choice_acc][index]

        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.C, self.h, self.w)
            )
        
        sample = sample.coalesce().to_dense()

        img_list = []
        for histo in sample:
            c = histo[0, :, :]
            img = torch.stack([c, c, c], dim=0)
            img_list.append(img)
        
        img_list = torch.stack(img_list, dim=0)

        sample = img_list[0]

        return sample, target

    def create_data(self, events):
        events['t'] -= events['t'][0]

        coords, feats = self.accumulate_events(events)

        return coords, feats

    def accumulate_events(self, events):
        dic_histo = {}

        total_slices = self.T
        events_micros = events['t'][-1]
        slice_interval = events_micros // total_slices
        
        for event in events:
            ts, x, y, p = event

            slice_index = (ts % events_micros) // slice_interval

            # Check if slice_index is within bounds
            if 0 <= slice_index < total_slices:

                if (p, slice_index, y, x) not in dic_histo:
                    dic_histo[(p, slice_index, y, x)] = 0

                if (dic_histo[(p, slice_index, y, x)] < 32767):
                    dic_histo[(p, slice_index, y, x)] += 1


        coordinates = torch.tensor([(key[1], key[0], key[2], key[3]) for key, value in dic_histo.items()], dtype=torch.int16)

        values = torch.tensor([value for key, value in dic_histo.items()], dtype=torch.int16)

        return coordinates, values