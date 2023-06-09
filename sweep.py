import time
import toml

import torch
import wandb


from src.utils.transformations import get_transformations, get_augmentations
from src.datasets.classification_dataset import get_dataloaders
from src.models.classifier import Classifier
from src.trainers.classification_trainer import create_trainer

config_path = "./configs/classification_sweep_config.toml"

base_config = toml.load(config_path)

sweep_config = base_config.pop("sweep_configuration")


def train(config: dict):

    model_name = config.model_names
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    use_scheduler = config.use_scheduler

    config = {key: value for key, value in base_config.items()}

    config["model"]["name"] = model_name
    config["optimizer"]["learning_rate"] = learning_rate
    config["dataloading"]["train_batch_size"] = batch_size
    config["dataloading"]["valid_batch_size"] = batch_size
    config["dataloading"]["test_batch_size"] = batch_size
    config["optimizer"]["use_scheduler"] = use_scheduler

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

    return best_valid_loss

def main():
    wandb.init(
        project=base_config["logging"]["project_name"],
    )

    score = train(wandb.config)
    
    # Add extra config to the wandb run

    wandb.log({"best_valid_loss": score})


sweep_id = wandb.sweep(
    sweep = sweep_config,
    project = base_config["logging"]["project_name"],
)

wandb.agent(sweep_id, function=main, count=50)