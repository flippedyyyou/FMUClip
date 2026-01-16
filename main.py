import torch
import torch.nn as nn
from torch.utils.data import random_split
from copy import deepcopy

import sys
import math

from load_dataset import UniversalDataset, UniversalDataLoader
from model import Model, CheckpointHelper
from expmgr import expmgr

from typing import *


def import_task_train_and_test(
    task: str,
) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """
    Dynamically imports the train and test functions from the corresponding task module.

    Args:
        task (str): Task name (e.g., "zero_shot", "image_retrieval").

    Returns:
        Tuple containing the train and test functions.
    """
    sys.path.append("tasks")
    if task == "zero_shot":
        from tasks.zero_shot import train, test
    elif task == "image_retrieval":
        from tasks.image_retrieval import train, test
    return train, test


def main() -> None:
    """
    Main experiment runner.

    Loads settings from the ExperimentManager, prepares the dataset, dataloaders, model,
    optimizer, and runs training (optionally testing).
    """
    with expmgr.ExperimentManager() as settings:

        # Load experiment configuration
        mode = settings["mode"]
        task = settings["task"]
        train, test = import_task_train_and_test(task)

        # Set seed and device
        torch.manual_seed(settings["misc"]["seed"])
        device = torch.device(settings["misc"]["device"])

        # Load training hyperparameters
        batch_size: int = settings[mode]["batch_size"]
        learning_rate = settings[mode]["learning_rate"]

        # Dataloader parameters
        shuffle: bool = settings["dataloader"]["shuffle"]
        num_workers: int = settings["dataloader"]["num_workers"]

        # Dataset loading
        dataset_name: str = settings["dataset"]["name"]
        dataset_info = settings["dataset_infos"][dataset_name]

        dataset = UniversalDataset(dataset_name, **dataset_info)

        # Set forget rule for filtering out forget set samples
        dataset.set_forget_rule(eval(settings["dataset"]["forget_rule"]))

        # Split dataset into training and testing (70% train / 30% test)
        print("Total dataset size:", len(dataset))
        train_num_samples = math.floor(0.7 * len(dataset))
        train_test_num_samples = [train_num_samples, len(dataset) - train_num_samples]
        train_set, test_set = random_split(dataset, train_test_num_samples)

        # Create data loaders
        train_loader = UniversalDataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        test_loader = UniversalDataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        # Initialize model and optimizer
        model = Model().to(device)
        model_ori = deepcopy(model)  # Save original model for possible comparison
        weight_decay = float(settings[mode]["weight_decay"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=weight_decay,
        )

        # Optional: alternative version of optimizer (commented out)
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=learning_rate, betas=(
        #         0.9, 0.98), weight_decay=settings[mode]["weight_decay"])

        # Set up checkpoint helper to save best models
        checkpoint_helper = CheckpointHelper()

        # Begin training
        train(model, train_loader, optimizer, settings, checkpoint_helper, test_loader)

        # Optional testing (commented out)
        # test(model, test_loader, settings)


if __name__ == "__main__":
    # Entry point of the script
    main()
