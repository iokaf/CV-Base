"""This module contains the definition of an image classifier model"""
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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

    callbacks = []

    if config["training"]["save_top_k"] > 0:
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

        callbacks.append(checkpoint_callback)

    if config["training"]["stopping_patience"] > 0:
        early_stopping_callback = EarlyStopping(
            monitor="valid_loss",
            patience=config["training"]["stopping_patience"],
            mode="min",
        )
        callbacks.append(early_stopping_callback)


    trainer = pl.Trainer(
        accelerator=config["training"]["device"],
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"]["precision"],
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    return trainer, checkpoint_callback
        
