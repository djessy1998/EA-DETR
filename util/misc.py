# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import cv2
import numpy as np
import time
from collections import defaultdict, deque
import datetime
import copy
import pickle
from packaging import version
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torchvision.ops import roi_align

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)

    targets = [item[1] for item in batch]
    #batch[0] = nested_tensor_from_tensor_list(batch[0])
    return [samples, targets]


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    return maxes



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # [[t, c, h, w], [t, c, h, w]]
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False

    else:
        raise ValueError('not supported')
    
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def filter_small_objects(detr_output, normalized_area_threshold=0.003335):
    """
    Filters the small objects from the DETR output.

    Parameters:
    - detr_output: dict, output of DETR containing 'pred_boxes' and 'pred_logits'
    - normalized_area_threshold: float, area threshold to consider an object as small

    Returns:
    - dict: filtered DETR output containing only small objects
    """
    pred_boxes = detr_output['pred_boxes']
    pred_logits = detr_output['pred_logits']

    # Calculate widths and heights in the normalized space
    widths = pred_boxes[..., 2]
    heights = pred_boxes[..., 3]

    # Calculate areas
    areas = widths * heights

    # Get indices of small objects
    small_object_indices = (areas < normalized_area_threshold).nonzero(as_tuple=True)[1]

    # Filter boxes and logits by small object indices
    filtered_pred_boxes = pred_boxes[:, small_object_indices, :]
    filtered_pred_logits = pred_logits[:, small_object_indices, :]

    return {
        'pred_boxes': filtered_pred_boxes,
        'pred_logits': filtered_pred_logits
    }


def get_small_bboxes(targets):

    targets_small = copy.deepcopy(targets)

    for annotations in targets_small:
        boxes = annotations['boxes']
        labels = annotations['labels']

        # Calculate areas
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        # Get indices of small objects
        small_object_indices = (areas < 0.003335).nonzero(as_tuple=True)[0]

        # Filter boxes and labels by small object indices
        annotations['boxes'] = boxes[small_object_indices]
        annotations['labels'] = labels[small_object_indices]

    # Return the list of small object bounding boxes
    return targets_small

def create_pos_neg_teacher(outputs_t, results_t, ind_t, ind_neg_t):
    orig_size_tensor = torch.tensor((480, 640), device='cuda')

    results_pos_t = []
    results_neg_t = []
    for idx_batch, batch in enumerate(outputs_t['pred_boxes']):
        results_t[idx_batch]['boxes'] = batch
        results_t[idx_batch]['orig_size'] = orig_size_tensor

        results_pos_t.append({'boxes': results_t[idx_batch]['boxes'][ind_t[idx_batch][0]],
                            'labels': results_t[idx_batch]['labels'][ind_t[idx_batch][0]],
                            'orig_size': results_t[idx_batch]['orig_size']
                            })
        
        results_neg_t.append({'boxes': results_t[idx_batch]['boxes'][ind_neg_t[idx_batch][0]],
                            'labels': results_t[idx_batch]['labels'][ind_neg_t[idx_batch][0]],
                            'orig_size': results_t[idx_batch]['orig_size']
                            })
        
    return (results_pos_t, results_neg_t)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def batch_tensors(tensors):
    # Determine the maximum size in each dimension
    max_size = tensors[0].size()
    for tensor in tensors[1:]:
        max_size = tuple(max(current, new) for current, new in zip(max_size, tensor.size()))
    
    # Pad each tensor to this max size
    padded_tensors = []
    for tensor in tensors:
        padding = [(0, max_dim - cur_dim) for cur_dim, max_dim in zip(tensor.size(), max_size)]
        # Flatten the padding list which is needed by pad function
        pad_config = [item for sublist in reversed(padding) for item in sublist]  # Reverse the padding dimensions
        padded_tensor = torch.nn.functional.pad(tensor, pad_config)
        padded_tensors.append(padded_tensor)
    
    # Stack all padded tensors along a new batch dimension
    batched_tensor = torch.stack(padded_tensors)
    return batched_tensor

def filter_with_events(results_t, ind_neg_t, image_events):
    idx_neg = [t[0] for t in ind_neg_t]
    result = []
    for idx_batch, batch in enumerate(idx_neg):
        image = image_events.tensors[idx_batch]
        batch_mask = torch.zeros(len(batch), dtype=torch.bool)
        for idx, indices in enumerate(batch):
            boxe = results_t[idx_batch]['boxes'][indices]
            x1 = int(boxe[0].item())
            y1 = int(boxe[1].item())
            x2 = int(boxe[2].item())
            y2 = int(boxe[3].item())

            if(torch.any(image[0, y1:y2, x1:x2])):
                batch_mask[idx] = True

        result.append((batch[batch_mask], torch.tensor(-1)))

    return result

def scale_bounding_boxes(bboxes, original_height, original_width, feature_map_height, feature_map_width):
    scaled_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(max(0, x_min / original_width * feature_map_width))
        x_max = int(max(0, x_max / original_width * feature_map_width))
        y_min = int(max(0, y_min / original_height * feature_map_height))
        y_max = int(max(0, y_max / original_height * feature_map_height))
        scaled_bboxes.append([x_min, y_min, x_max, y_max])

    return scaled_bboxes

def extract_features(feature_map, scaled_bboxes):
    features = []
    for bbox in scaled_bboxes:
        x_min, y_min, x_max, y_max = bbox
        roi_features = feature_map[:, y_min:y_max, x_min:x_max]  # Extract region of interest
        # Optionally, you can perform pooling to get a fixed-size feature vector for each bounding box
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(roi_features, (1, 1)).squeeze(-1).squeeze(-1)
        features.append(pooled_features)
    return torch.stack(features)

def compute_bounding_box_mse_loss(results, feature_map1, feature_map2):
    mse_loss = nn.MSELoss()
    total_loss = 0
    batch_size = feature_map1.shape[0]
    orig_height, orig_width = 480, 640

    for i in range(batch_size):
        bboxes = results[i]['boxes']

        clamped_boxes = torch.zeros_like(bboxes)
        clamped_boxes[:, 0] = torch.clamp(bboxes[:, 0], min=0, max=orig_width)  # x_min
        clamped_boxes[:, 1] = torch.clamp(bboxes[:, 1], min=0, max=orig_height)  # y_min
        clamped_boxes[:, 2] = torch.clamp(bboxes[:, 2], min=0, max=orig_width)  # x_max
        clamped_boxes[:, 3] = torch.clamp(bboxes[:, 3], min=0, max=orig_height)  # y_max

        scores = results[i]['scores']
        scores = scores > 0.5
        bboxes = clamped_boxes[scores]

        batch_indices = torch.zeros((bboxes.size(0), 1), device='cuda')  # Assuming all boxes are in batch 0
        rois = torch.cat([batch_indices, bboxes], dim=1)  # Shape: [num_boxes, 5]

        roi_te = roi_align(feature_map2, rois, (4, 4), spatial_scale=0.0625, sampling_ratio=-1)
        roi_st = roi_align(feature_map1, rois, (4, 4), spatial_scale=0.0625, sampling_ratio=-1)
        
        loss = mse_loss(roi_st, roi_te)

        total_loss += loss

    total_loss /= batch_size

    return total_loss



def get_negative_idx(tensor_list):
    all_numbers = torch.arange(0, 100)  # All numbers from 0 to 99
    result = []
    for tensor in tensor_list:
        mask = torch.ones(100, dtype=torch.bool)  # Create a mask of size 100, initialized to True
        mask[tensor] = False  # Set False at indices present in the tensor
        result.append((all_numbers[mask], torch.tensor(-1)))  # Append the numbers not present in the tensor
    return result

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
