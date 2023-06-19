from typing import Dict

import time
import toml

import torch
import wandb
import gc

from src.utils.transformations import get_transformations, get_augmentations
from src.datasets.classification_dataset import get_dataloaders
from src.models.classifier import Classifier
from src.trainers.classification_trainer import create_trainer

config_path = "./configs/classification_sweep_config.toml"

base_config = toml.load(config_path)

sweep_config = base_config.pop("sweep_configuration")


def train(config: dict, val_dict: Dict):

    model_name = wandb.config.model_names
    learning_rate = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    use_scheduler = wandb.config.use_scheduler


    print(f"Model name: {model_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Use scheduler: {use_scheduler}")

    config = {key: value for key, value in base_config.items()}

    config["model"]["name"] = model_name
    config["optimizer"]["learning_rate"] = learning_rate
    config["dataloading"]["train_batch_size"] = batch_size
    config["dataloading"]["valid_batch_size"] = batch_size
    config["dataloading"]["test_batch_size"] = batch_size
    config["optimizer"]["use_scheduler"] = use_scheduler

    transformations = get_transformations(config=config)
    augmentations = get_augmentations(config=config)

    loaders = get_dataloaders(
        config=config,
        transforms=transformations,
        augmentations=augmentations,
    )
    

    config["train_dataloader_length"] = len(loaders["train_dataloader"])
    config["valid_dataloader_length"] = len(loaders["valid_dataloader"])

    if loaders["test_dataloader"] is not None:
        config["test_dataloader_length"] = len(loaders["test_dataloader"])
    else:
        config["test_dataloader_length"] = 0


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

    # Get the best validation loss from the classifier 
    best_valid_loss = checkpoint_callback.best_model_score

    del classifier, loaders, trainer, checkpoint_callback
    
    time.sleep(2)
    torch.cuda.empty_cache()
    gc.collect()
    val_dict["best_valid_loss"] = best_valid_loss.item()

    return val_dict

import multiprocessing as mp



def main():
    wandb.init(
        project=base_config["logging"]["project_name"],
    )


    manager = mp.Manager()
    return_dict = manager.dict()

    # Create a threading process that calls the train function with the config
    # and the values dict. The values dict is shared between all the processes
    # and is used to store the best validation loss for each process.

    wandb.log(base_config)
    
    p = mp.Process(target=train, args=(base_config, return_dict))
    p.start()
    p.join()

    
    
    p.terminate()
    # Log the best validation loss for each process
    wandb.log({"best_valid_loss": return_dict["best_valid_loss"]})

    return return_dict["best_valid_loss"]



sweep_id = wandb.sweep(
    sweep = sweep_config,
    project = base_config["logging"]["project_name"],
)

wandb.agent(sweep_id, function=main, count=150)