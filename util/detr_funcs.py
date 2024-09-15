import torch
import cv2
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy

dic_labels = {
    0: 'pedestrian',
    1: 'rider',
    2: 'car',
    3: 'bus',
    4: 'truck',
    5: 'bicycle',
    6: 'motorcycle',
    7: 'train',
    8: 'background'
}


'''
results, targets = scale_for_map_computation(image_events, image_rgb, outputs, targets, self.postprocessors, self.index_im)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0

bboxes_scaled = results[0]['boxes'][keep]


image_viz = cv2.normalize(image_rgb.tensors[0].permute(1,2,0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

image_viz_ev = cv2.normalize(image_events.tensors[0].permute(1,2,0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
image_viz_ev = cv2.convertScaleAbs(image_viz_ev, alpha=5.0, beta=75)

for hook in hooks:
    hook.remove()
'''

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

def format_targets(targets, index):
    result = []
    for i, target in enumerate(targets):
        boxes = target['boxes'].tolist()
        labels = target['labels'].tolist()
        box_strings = ", ".join([f"[{', '.join(map(str, box))}]" for box in boxes])
        result.append(f"{index}: {box_strings} -> {labels}")
    return "\n".join(result)

def get_last_layer_encoder_attn(enc_att_weights):
        

        



        conv_features = conv_features[0]
        enc_att_weights = enc_att_weights[0]
        dec_attn_weights = dec_attn_weights[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        viz_enc = enc_att_weights.mean(dim=1).view(h, w)
        #viz_dec = dec_attn_weights[0][keep]

        #nb_boxes = viz_dec.shape[0]

        #viz_dec = viz_dec.view(nb_boxes, h, w).mean(dim=0)

        viz_enc = cv2.normalize(viz_enc.detach().cpu().numpy(), None, 0, 1, norm_type=cv2.NORM_MINMAX)

        import pdb
        pdb.set_trace()

        resized_image = cv2.resize(viz_enc, (640, 480))
        mask_att = resized_image > 0.2
        mask_att = np.stack([mask_att]*3, axis=-1)


        #plt.imshow(image_viz)
        #plt.show()

        #viz_deco_resized = cv2.resize(viz_dec.detach().cpu().numpy(), (640, 480))

        #plt.imshow(resized_image)
        #plt.imshow(viz_deco_resized, alpha=0.5)
        plt.imshow(image_viz_ev)
        plt.show()

        '''
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        
        CLASSES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'background']

        import pdb
        pdb.set_trace()

        fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        colors = COLORS * 100
        for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
            ax = ax_i[0]
            dec_viz = dec_attn_weights[0, idx].view(h, w).detach().cpu().numpy()
            ax.imshow(dec_viz)
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(image_viz)
            ax.add_patch(plt.Rectangle((xmin.cpu(), ymin.cpu()), xmax.cpu() - xmin.cpu(), ymax.cpu() - ymin.cpu(),
                                    fill=False, color='blue', linewidth=3))
            ax.axis('off')
            ax.set_title(CLASSES[probas[idx].argmax()])
        fig.tight_layout()

        plt.show()

        # output of the CNN
        f_map = conv_features['0']
        print("Encoder attention:      ", enc_att_weights[0].shape)
        print("Feature map:            ", f_map.tensors.shape)

        # get the HxW shape of the feature maps of the CNN
        shape = f_map.tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        sattn = enc_att_weights[0].reshape(shape + shape).cpu().numpy()
        print("Reshaped self-attention:", sattn.shape)

        # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
        fact = 16

        # let's select 4 reference points for visualization
        idxs = [(200, 50), (280, 400), (200, 600), (440, 100),]

        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(30 * 0.7, 10 * 0.7))
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        axs = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[0, -1]),
            fig.add_subplot(gs[1, -1]),
        ]

        # for each one of the reference points, let's plot the self-attention
        # for that point
        for idx_o, ax in zip(idxs, axs):
            idx = (idx_o[0] // fact, idx_o[1] // fact)
            ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'self-attention{idx_o}')

        # and now let's add the central image, with the reference points as red circles
        fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        fcenter_ax.imshow(image_viz)
        
        for (y, x) in idxs:
            scale = image_viz.shape[0] / image_rgb.tensors[0].shape[-2]
            x = ((x // fact) + 0.5) * fact
            y = ((y // fact) + 0.5) * fact
            fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
            fcenter_ax.axis('off')
        
        plt.show()
        '''


def add_hooks_detr(model, features, enc_attn,  dec_attn):
    hooks_student = [

    ] 

def scale_for_map_computation(image, image_rgb, outputs, targets, postprocessors, index):
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['bbox'](outputs, orig_target_sizes)
    
    for idx, result in enumerate(results):
        results[idx]['boxes'] = result['boxes'].to(torch.float32)
        results[idx]['scores'] = result['scores'].to(torch.float16)

    img_h_tensor = torch.tensor([int(image.tensors.shape[2])], dtype=torch.float32)
    img_w_tensor = torch.tensor([int(image.tensors.shape[3])], dtype=torch.float32)

    # Create the scale factor tensor
    scale_fct = torch.stack([img_w_tensor, img_h_tensor, img_w_tensor, img_h_tensor], dim=1).squeeze(0)

    for idx, t in enumerate(targets):
        # Apply the scaling
        gt = box_cxcywh_to_xyxy(t['boxes'].clone().detach().cpu() * scale_fct)
        targets[idx]['boxes'] = gt

    image = cv2.normalize(image.tensors[0].permute(1,2,0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.normalize(image_rgb.tensors[0].permute(1,2,0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    for i in range(len(targets[0]['boxes'])):
        bbx_target = targets[0]['boxes'][i]
        cv2.rectangle(image, (int(bbx_target[0]), int(bbx_target[1])), (int(bbx_target[2]), int(bbx_target[3])), [0,0,255], 2, cv2.LINE_AA)
        cv2.rectangle(image_rgb, (int(bbx_target[0]), int(bbx_target[1])), (int(bbx_target[2]), int(bbx_target[3])), [0,0,255], 2, cv2.LINE_AA)

    for i in range(len(results[0]['boxes'])):
        if(results[0]['scores'][i] > 0.5):
            labels = results[0]['labels'][i]
            bbx_output = results[0]['boxes'][i]
            name = dic_labels[int(labels)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (255, 0, 0) # Green in BGR format
            text_thickness = 2
            bottom_left_text = (int(bbx_output[0]) + 10, int(bbx_output[1]) - 10) # Position just above the rectangle

            # Add text
            cv2.putText(image, name, bottom_left_text, font, font_scale, text_color, text_thickness)
            cv2.rectangle(image, (int(bbx_output[0]), int(bbx_output[1])), (int(bbx_output[2]), int(bbx_output[3])), [255,0,0], 2, cv2.LINE_AA)
            cv2.rectangle(image_rgb, (int(bbx_output[0]), int(bbx_output[1])), (int(bbx_output[2]), int(bbx_output[3])), [255,0,0], 2, cv2.LINE_AA)
    

    #cv2.imshow("event", image)
    #cv2.imshow("rgb", image)
    #cv2.imwrite(f"/home/djessy/Bureau/CodeThese/distill_resnet/HardEventDSEC-DET/seq_tunnels/rgb/rgb_tunnels_{index}.png", image_rgb)
    #cv2.waitKey(0)


    return (results, targets)