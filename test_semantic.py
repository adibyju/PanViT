"""
Script for semantic inference with pre-trained models
Modified for Spatio-Temporal Swin Transformer
"""

import argparse
import json
import os
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src import utils
from src.dataset import PASTIS_Dataset
from train_semantic import iterate, overall_performance, save_results, prepare_output
from train_semantic import SpatioTemporalSwinModel  # Import the model we defined for training

# Arguments Parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the dataset is stored.",
)
parser.add_argument(
    "--res_dir",
    default="./inference_swin",
    type=str,
    help="Path to directory where results are written."
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of test metrics",
)

def main(config):
    # Define the fold sequence for cross-validation
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)
    prepare_output(config)

    # Load the pre-trained Spatio-Temporal Swin Transformer model
    model = SpatioTemporalSwinModel(num_classes=config.num_classes)
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    # Adjust fold sequence for specific folds
    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Define the test dataset
        dt_test = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="semantic",
            sats=["S2"],
            folds=test_fold,
        )
        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # Load the model weights for the current fold
        sd = torch.load(
            os.path.join(config.weight_folder, "Fold_{}".format(fold + 1), "model.pth.tar"),
            map_location=device,
        )
        model.load_state_dict(sd["state_dict"])

        # Define the loss function
        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Perform inference
        print("Testing fold {}...".format(fold + 1))
        model.eval()
        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=None,  # No optimizer during testing
            mode="test",
            device=device,
        )

        # Print the results for the current fold
        print(
            "Loss {:.4f}, Acc {:.2f}, IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )

        # Save the results
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)

    # If running all folds, calculate overall performance
    if config.fold is None:
        overall_performance(config)

if __name__ == "__main__":
    test_config = parser.parse_args()

    # Load the configuration from the training process (to match model architecture and hyperparameters)
    with open(os.path.join(test_config.weight_folder, "conf.json")) as file:
        model_config = json.loads(file.read())

    # Merge model config with test config (command-line arguments)
    config = {**model_config, **vars(test_config)}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold  # Override fold if provided during testing

    pprint.pprint(config)
    main(config)
