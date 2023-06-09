"""This module contains the definition of an image classifier model"""

from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import timm
import torch

from src.loggers import ClassificationMetricMonitor


class Classifier(pl.LightningModule):
    """Implementation of the classifier class."""

    def __init__(
            self, 
            config: Dict,
            wandb: Any = None):
        """Initializes the classifier.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.model = timm.create_model(
            config["model"]["name"],
            pretrained=config["model"]["pretrained"],
            num_classes=config["model"]["number_of_classes"],
        )

        self.name = config["model"]["name"]

        self.use_scheduler = config["optimizer"]["use_scheduler"]

        self.wandb = wandb

        self.set_loss_and_activation(config["model"]["mutually_exclusive"])
        self.create_loggers(config["model"]["number_of_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Argsuments:
        ----------
            x (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor.
        """
        return self.model(x)
    

    def step(
            self, 
            batch: Dict[str, torch.Tensor], 
            batch_idx: int
        ) -> Dict[str, torch.Tensor]:
        """Performs a single step in the training loop.

        Arguments:
        ----------
        batch (dict): 
            Batch of data.

        batch_idx (int):
            Index of the batch.

        Returns:
        --------
        dict: Dictionary containing the loss and the predictions.
        """
        x, y = batch["image"], batch["label"]
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        predictions = self.activation(y_hat)

        output =  {
            "loss": loss, 
            "logits": y_hat,
            "predictions": predictions,
            "targets": y,
            "batch_idx": batch_idx,
        }

        return output
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ):
        """Training step.

        Arguments:
        ----------
        batch (dict): 
            Batch of data.
        batch_idx (int): 
            Index of the batch.
        Returns:
        --------
        dict: Dictionary containing the loss and the predictions.
        """
        result = self.step(batch, batch_idx)

        self.log(
            "train_loss", 
            result["loss"], 
            on_step=True, 
            on_epoch=True,
            prog_bar=True
        )
        
        self.train_metric_monitor.update(result)

        if self.wandb:
            self.wandb.log({
                "train_loss": result["loss"], 
                "epoch": self.current_epoch, 
                "batch": batch_idx, 
                "step": self.global_step, 
                "mode": "train"
            })

        return result   

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Validation step.
        
        Arguments:
        ----------
        batch (dict):
            Batch of data.
        batch_idx (int):
            Index of the batch.
        Returns:
        --------
        dict: Dictionary containing the loss and the predictions.
        """

        result = self.step(batch, batch_idx)

        self.log(
            "valid_loss", 
            result["loss"], 
            on_step=True, 
            on_epoch=True,
            prog_bar=True    
        )
        self.valid_metric_monitor.update(result)

        if self.wandb:
            self.wandb.log({
                "valid_loss": result["loss"], 
                "epoch": self.current_epoch, 
                "batch": batch_idx, 
                "step": self.global_step, 
                "mode": "valid"
            })

        return result

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Testing step.
        
        Arguments:
        ----------
        batch (dict):
            Batch of data.
        batch_idx (int):
            Index of the batch.
        Returns:
        --------
        dict: Dictionary containing the loss and the predictions.
        """

        result = self.step(batch, batch_idx)

        self.log("test_loss", result["loss"], on_step=True, on_epoch=True)
        if self.wandb:
            self.wandb.log({
                "test_loss": result["loss"], 
                "epoch": self.current_epoch, 
                "batch": batch_idx, 
                "step": self.global_step, 
                "mode": "test"
            })
        self.test_metric_monitor.update(result)

        return result
    
    def on_train_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]=None):
        """Training epoch end hook.

        Arguments:
        ----------
        outputs (list):
            List of dictionaries containing the loss and the predictions.
        """
        epoch_metrics = self.train_metric_monitor.compute()
        self.log_dict(epoch_metrics)

        if self.wandb:
            self.wandb.log(epoch_metrics)

        self.train_metric_monitor.reset()

    def on_validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]=None):
        """Validation epoch end hook.

        Arguments:
        ----------
        outputs (list):
            List of dictionaries containing the loss and the predictions.
        """
        epoch_metrics = self.valid_metric_monitor.compute()
        self.log_dict(epoch_metrics)

        if self.wandb:
            self.wandb.log(epoch_metrics)

        self.valid_metric_monitor.reset()


    def on_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]=None):
        """Test epoch end hook.

        Arguments:
        ----------
        outputs (list):
            List of dictionaries containing the loss and the predictions.
        """
        epoch_metrics = self.test_metric_monitor.compute()
        self.log_dict(epoch_metrics)

        if self.wandb:
            self.wandb.log(epoch_metrics)

        self.test_metric_monitor.reset()

    def create_loggers(self, num_classes: int):
        """Creates the loggers for the model."""

        if num_classes > 1:
            task = "multiclass"
            num_classes = num_classes
        else:
            task = "binary"
            num_classes = None

        self.train_metric_monitor = ClassificationMetricMonitor(
            task = task,
            threshold = 0.5,
            num_classes = num_classes,
            prefix = "train"
        )

        self.valid_metric_monitor = ClassificationMetricMonitor(
            task = task,
            threshold = 0.5,
            num_classes = num_classes,
            prefix = "valid"
        )

        self.test_metric_monitor = ClassificationMetricMonitor(
            task = task,
            threshold = 0.5,
            num_classes = num_classes,
            prefix = "test"
        )

    def set_loss_and_activation(self, mutually_exclusive: bool):
        """Sets the loss function.

        Arguments:
        ----------
        mutually_exclusive (bool):
            Whether the classes are mutually exclusive or not.
        """
        if mutually_exclusive:
            self.criterion = torch.nn.CrossEntropyLoss()
            self.activation = torch.nn.Softmax(dim=1)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.activation = torch.nn.Sigmoid()

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Configures the optimizer."""

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"],
        )

        if not self.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["max_epochs"],
            eta_min=self.config["optimizer"]["eta_min"],
        )

        return [optimizer], [scheduler]
    