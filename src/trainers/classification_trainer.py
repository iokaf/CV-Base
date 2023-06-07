"""This module contains the definition of an image classifier model"""
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def create_trainer(config: Dict) -> pl.Trainer:
    """ Creates a pytorch lightning trainer.

    Arguments:
    ----------
        config (dict): Configuration dictionary.

    Returns:
    --------
        pl.Trainer: Pytorch lightning trainer.
    """

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=config["training"]["checkpoint_directory"],
        filename=(
            config["model"]["name"] +
            date +
            "-{epoch:02d}-{valid_loss:.5f}"
            ),
        save_top_k=config["training"]["save_top_k"],
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=config["training"]["device"],
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"]["precision"],
        callbacks=[checkpoint_callback],
    )

    return trainer, checkpoint_callback
        
