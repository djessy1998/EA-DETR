import argparse
import pytorch_lightning as L
import torch
import hdf5plugin

from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule.datamodule_dsec import DSECDataModule
from datamodule.datamodule_gen1 import GEN1DataModule
from datamodule.datamodule_1mpx import MPXDataModule
from datamodule.datamodule_hedsec import HEDsecDataModule

from models_lightning.detr_lightning import Detr
from models_lightning.detr_distill import DetrDistill
from models_lightning.detr_distill_frozen_decoder import DetrDistillFD

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Event-based object detection with specialized DETR')

    #### DETR ####
    # Training
    parser.add_argument('-b', default=1, type=int)
    parser.add_argument('-wd', default=1e-4, type=float, help='weight decay used')
    parser.add_argument('-epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('-lr', default=1e-4, type=float)
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-device', default=0, type=int, help='device')

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--lr_drop', default=200, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Transformer
    parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Backbone
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_false',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    #### End DETR ####
    
    # Distillation
    parser.add_argument('-distillation', action='store_true', help='DETR Distillation')
    parser.add_argument('-features', action='store_true', help='distillation backbone DETR with MSE loss')
    parser.add_argument('-ddfd', action='store_true')
    parser.add_argument('-sparse_loss_bcb', action='store_true')
    parser.add_argument('-distill_logits', action='store_true', help='distillation logits DETR')
    parser.add_argument('-teacher_detr_root', default='weights/dsec-det/weight_best_model_detr_dsec_rgb/-epoch=50-val_AP_IoU_0_5_to_0_95=0.523131.ckpt', help='Root of your teacher DETR for distillation')

    # Testing or training
    parser.add_argument('-train', action='store_true', help='training mode')

    #### Testing ####
    parser.add_argument('-test', action='store_true', help='testing mode')
    parser.add_argument('-p', action='store_true', help='whether or not use a pretrained DETR')
    parser.add_argument('-ckpt_pretrained_detr', default='weights/dsec-det/weight_best_model_detr_dsec_events/baseline/-epoch=42-val_AP_IoU_0_5_to_0_95=0.308461.ckpt', help='Checkpoint pretrained DETR')
    #### End Testing ####

    parser.add_argument('--event', dest='event', action='store_true')
    
    #### Datasets ####
    dataset_choice = parser.add_mutually_exclusive_group(required=True)
    
    dataset_choice.add_argument('-dsec_det', dest='datamodule_factory', action='store_const',
                       const=lambda args: DSECDataModule(args.root_dsec, args.b), help='Train or test on dsec-det dataset')
    dataset_choice.add_argument('-gen1', dest='datamodule_factory', action='store_const',
                       const=lambda args: GEN1DataModule(args, args.b), help='Train or test on gen1 dataset')
    dataset_choice.add_argument('-1mpx', dest='datamodule_factory', action='store_const',
                       const=lambda args: MPXDataModule(args.root_1mpx, args.b), help='Train or test on 1mpx dataset')
    dataset_choice.add_argument('-hedsec', dest='datamodule_factory', action='store_const',
                       const=lambda args: HEDsecDataModule(args.root_hedsec, args.b), help='Test on Hard Event dataset')

    parser.add_argument('-root_dsec', default='/home/djessy/Bureau/CodeThese/dsec-det/DSEC_MERGED', type=Path)
    parser.add_argument('-root_gen1', default='/media/djessy/LaCie/GEN1', type=Path)
    parser.add_argument('-root_1mpx', default='', type=Path)
    parser.add_argument('-root_hedsec', default='/home/djessy/Bureau/CodeThese/distill_resnet/HardEventDSEC-DET', type=Path)

    parser.add_argument('--dataset_file', default='dsec')

    #DSEC

    #GEN1
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-image_shape', default=(240,304), type=tuple, help='spatial resolution of events')
    parser.add_argument('-T', default=5, type=int, help='number of total histograms')

    parser.add_argument('-acc_augmentation', default=[100000], type=list, help='time of different accumulation in µs')

    #### End Datasets ####

    args = parser.parse_args()

    datamodule = args.datamodule_factory(args)

    logger = TensorBoardLogger("profiling_tb", name="detr_distill")
    callbacks=[]
    if args.save_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='val_AP_IoU_0_5_to_0_95',
            dirpath=f"ckpt-od-{args.backbone}-{args.version}/",
            filename="" + "-{epoch:02d}-{val_AP_IoU_0_5_to_0_95:04f}",
            save_top_k=1,
            mode='max',
        )
        callbacks.append(ckpt_callback)


    trainer = L.Trainer(
        accelerator='gpu',
        logger=logger,
        devices=1,
        max_epochs=1,
        callbacks=callbacks
    )


    # Using native Detr model by default
    model = Detr(args=args)

    if args.train:
        # You want to train distillation between pretrained teacher and student
        if args.distillation:
            model = DetrDistill(args=args)

        if args.ddfd:
            model = DetrDistillFD(args=args)

        # Else you train a normal DETR
        trainer.fit(model=model, datamodule=datamodule)

    if args.test:
        if args.p:
            model = Detr.load_from_checkpoint(args=args, checkpoint_path=args.ckpt_pretrained_detr)
            model.eval()
        else:
            model = DetrDistill.load_from_checkpoint(args=args, checkpoint_path="weights/dsec-det/weight_best_model_detr_dsec_events/hungarian_matchin_resnet_distill/-epoch=45-val_AP_IoU_0_5_to_0_95=0.425923.ckpt")
            model.eval()

        trainer.test(model=model, datamodule=datamodule)

    #Choice datamodule


if __name__ == '__main__':
    main()

