"""This module is used to train the model with a given configuration file."""
import toml

import wandb

from src.utils.transformations import get_transformations, get_augmentations
from src.datasets.classification_dataset import get_dataloaders
from src.models.classifier import Classifier
from src.trainers.classification_trainer import create_trainer

config_path = "./configs/classification_config.toml"

config = toml.load(config_path)


transformations = get_transformations(config=config)
augmentations = get_augmentations(config=config)

data_path = config["data"]["annotations_path"]

loaders = get_dataloaders(
    data_path=data_path,
    config=config,
    transforms=transformations,
    augmentations=augmentations,
)

config["train_dataloader_length"] = len(loaders["train_dataloader"])
config["valid_dataloader_length"] = len(loaders["valid_dataloader"])
config["test_dataloader_length"] = len(loaders["test_dataloader"])


wandb.init(
    project=config["logging"]["project_name"],
    config=config,
    reinit=True
)

classifier = Classifier(
    config=config,
    wandb=wandb
    )

trainer, checkpoint_callback = create_trainer(config=config)

trainer.fit(
    classifier, 
    loaders["train_dataloader"], 
    loaders["valid_dataloader"]
)

print(f"Best validation loss: {checkpoint_callback.best_model_score}")


best_path = checkpoint_callback.best_model_path

classifier = Classifier.load_from_checkpoint(
    checkpoint_path=best_path,
    config=config,
    wandb=None
)

trainer.test(
    classifier,
    loaders["test_dataloader"]
)
