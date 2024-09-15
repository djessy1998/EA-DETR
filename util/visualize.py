import copy
import torch
import cv2
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy
from util.detr_funcs import *

def visualize_pos_neg_boxes(image, image_rgb, targets, results_s, results_t, index):
    targets_vis = copy.deepcopy(targets)

    image = cv2.normalize(image.tensors[0].permute(1,2,0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.normalize(image_rgb.tensors[0].permute(1,2,0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.convertScaleAbs(image, alpha=5.0, beta=75)

    for i in range(len(targets_vis[0]['boxes'])):
        bbx_target = targets_vis[0]['boxes'][i]
        cv2.rectangle(image, (int(bbx_target[0]), int(bbx_target[1])), (int(bbx_target[2]), int(bbx_target[3])), [0,0,255], 2, cv2.LINE_AA)

    for i in range(len(results_s[0]['boxes'])):
        if(results_s[0]['scores'][i] > 0.5):
            labels = results_s[0]['labels'][i]
            bbx_output = results_s[0]['boxes'][i]
            name = dic_labels[int(labels)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 255, 0) # Green in BGR format
            text_thickness = 2
            bottom_left_text = (int(bbx_output[0]) + 10, int(bbx_output[1]) - 10) # Position just above the rectangle

            # Add text
            cv2.putText(image, name, bottom_left_text, font, font_scale, text_color, text_thickness)
            cv2.rectangle(image, (int(bbx_output[0]), int(bbx_output[1])), (int(bbx_output[2]), int(bbx_output[3])), [0,255,0], 2, cv2.LINE_AA)


    for i in range(len(results_t[0]['boxes'])):
        if(results_t[0]['scores'][i] > 0.5):
            labels = results_t[0]['labels'][i]
            bbx_output = results_t[0]['boxes'][i]
            name = dic_labels[int(labels)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 255, 0) # Green in BGR format
            text_thickness = 2
            bottom_left_text = (int(bbx_output[0]) + 10, int(bbx_output[1]) - 10) # Position just above the rectangle

            # Add text
            cv2.putText(image, name, bottom_left_text, font, font_scale, text_color, text_thickness)
            cv2.rectangle(image, (int(bbx_output[0]), int(bbx_output[1])), (int(bbx_output[2]), int(bbx_output[3])), [0,255,0], 2, cv2.LINE_AA)

    cv2.imshow("event", image)
    cv2.imshow("rgb", image_rgb)
    cv2.waitKey(0)