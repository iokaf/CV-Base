"""This module trains the recursive predictor model."""
import os
import toml

import matplotlib.pyplot as plt
import wandb

from src.datasets.lstm_classification_dataset import get_dataloaders
from src.utils.transformations import get_transformations, get_augmentations
from src.trainers.classification_trainer import create_trainer
from src.models.lstm_classifier import LSTMClassifier

config_path = "./configs/recurrent_config.toml"

config = toml.load(config_path)


transforms = get_transformations(config)
augmentations = get_augmentations(config)

loaders = get_dataloaders(
    config=config,
    transforms=transforms,
    augmentations=augmentations
)

config["train_dataloader_length"] = len(loaders["train_dataloader"])
config["valid_dataloader_length"] = len(loaders["valid_dataloader"])

if loaders.get("test_dataloader") is not None:
    config["test_dataloader_length"] = len(loaders["test_dataloader"])
else:
    config["test_dataloader_length"] = 0

wandb.init(
    project=config["logging"]["project_name"],
    config=config,
    reinit=True
)

model = LSTMClassifier(config, wandb=wandb)

trainer, checkpoint_callback = create_trainer(config)

trainer, checkpoint_callback = create_trainer(config=config)

trainer.fit(
    model, 
    loaders["train_dataloader"], 
    loaders["valid_dataloader"]
)

print(f"Best validation loss: {checkpoint_callback.best_model_score}")
