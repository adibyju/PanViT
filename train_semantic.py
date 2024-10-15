import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt

from src import utils, model_utils
from src.dataset import PASTIS_Dataset
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.learning.weight_init import weight_init
from transformers import SwinForImageClassification

# Arguments Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="swin", type=str, help="Type of architecture to use.")
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]", type=str)
parser.add_argument("--num_workers", default=8, type=int, help="Number of data loading workers")
parser.add_argument("--device", default="cuda", type=str, help="Device (cuda/cpu)")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--res_dir", default="./results", type=str, help="Folder to store results")
parser.add_argument("--dataset_folder", default="", type=str, help="Path to dataset folder")



parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

parser.set_defaults(cache=False)


list_args = ["encoder_widths", "decoder_widths", "out_conv"]

# Model: Spatio-Temporal Swin Transformer
class SpatialEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(SpatialEncoder, self).__init__()
        # Pretrained Swin Transformer for spatial encoding
        self.swin_transformer = SwinForImageClassification.from_pretrained(
            'microsoft/swin-base-patch4-window7-224', output_hidden_states=True
        )

    def forward(self, x):
        # Input shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Flatten time dimension
        spatial_features = self.swin_transformer(x).hidden_states[-1]  # Extract features
        return spatial_features.view(B, T, -1, H // 32, W // 32)  # Reshape

class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim=768, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super(TemporalEncoder, self).__init__()
        # Swin Transformer for temporal encoding
        self.temporal_swin = SwinForImageClassification.from_pretrained(
            'microsoft/swin-base-patch4-window7-224'
        )

    def forward(self, x):
        # Input shape: [B, T, D, H', W'] (spatial features)
        B, T, D, H, W = x.shape
        x = x.view(B, T, D, H * W)  # Flatten spatial dimensions
        return self.temporal_swin(x).logits  # Output temporally encoded features

# Full Spatio-Temporal Swin Model
class SpatioTemporalSwinModel(nn.Module):
    def __init__(self, num_classes):
        super(SpatioTemporalSwinModel, self).__init__()
        self.spatial_encoder = SpatialEncoder()  # Pre-trained spatial Swin Transformer
        self.temporal_encoder = TemporalEncoder()  # Pre-trained temporal Swin Transformer
        self.segmentation_head = nn.Conv2d(768, num_classes, kernel_size=1)

    def forward(self, x):
        spatial_features = self.spatial_encoder(x)  # [B, T, D, H', W']
        temporal_output = self.temporal_encoder(spatial_features)  # [B, H', W', D]
        return self.segmentation_head(temporal_output)

def iterate(model, data_loader, criterion, config, optimizer=None, mode="train", device=None):
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()
        if mode != "train":
            with torch.no_grad():
                out = model(x)
        else:
            optimizer.zero_grad()
            out = model(x)
        loss = criterion(out, y)
        if mode == "train":
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, y)
        loss_meter.add(loss.item())
        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )
    t_end = time.time()
    total_time = t_end - t_start
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }
    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics

def main(config):
    device = torch.device(config.device)
    dataset_args = dict(
        folder=config.dataset_folder,
        norm=True,
        target="semantic",
        sats=["S2"],
    )

    # Dataset
    dt_train = PASTIS_Dataset(**dataset_args, cache=config.cache)
    collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
    train_loader = data.DataLoader(dt_train, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # Model definition
    model = SpatioTemporalSwinModel(num_classes=config.num_classes).to(device)
    model.apply(weight_init)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    weights = torch.ones(config.num_classes, device=device).float()
    weights[config.ignore_index] = 0
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training loop
    for epoch in range(1, config.epochs + 1):
        print(f"EPOCH {epoch}/{config.epochs}")
        model.train()
        train_metrics = iterate(
            model, data_loader=train_loader, criterion=criterion, config=config, optimizer=optimizer, mode="train", device=device
        )
        print(f"Loss {train_metrics['train_loss']:.4f}, Accuracy {train_metrics['train_accuracy']:.2f}, IoU {train_metrics['train_IoU']:.2f}")

if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "").replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))
    assert config.num_classes == config.out_conv[-1]
    pprint.pprint(config)
    main(config)
