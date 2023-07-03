"""This module contains the definition of an image classifier model"""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import timm
import torch
import wandb

from src.loggers import BinaryClassificationMetricMonitor, MulticlassClassificationMetricMonitor


class BaseClassifier(pl.LightningModule):
    """Implementation of the classifier class."""

    def __init__(
            self,
            config: Dict,
            wandb: Any = None,
            sweep: bool = False):
        """Initializes the classifier.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__()
        self.config = config
        
        self.name = config["model"]["name"]
        self.labels = config["data"]["label_names"]
        self.use_scheduler = config["optimizer"]["use_scheduler"]

        self.wandb = wandb

        self.set_loss_and_activation(
            config["model"]["mutually_exclusive"],
            config["model"]["number_of_classes"]
        )
        self.create_loggers(
            num_classes=config["model"]["number_of_classes"],
            mutually_exclusive=config["model"]["mutually_exclusive"],
            label_names=config["data"]["label_names"]
        )

        self.model = timm.create_model(
            config["model"]["name"],
            pretrained=config["model"]["pretrained"],
            num_classes=config["model"]["number_of_classes"],
        )

        if self.config["model"]["load_model"]:
            cpt_path = self.config["model"]["model_checkpoint_path"]
            state_dict = torch.load(cpt_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {cpt_path}")

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
        current_epoch = self.current_epoch * 1.0

        epoch_metrics["epoch"] = current_epoch

        conf_mat_key = list(filter(lambda x: "confusion_matrix" in x, epoch_metrics.keys()))
        conf_mat = epoch_metrics.pop(conf_mat_key[0])
        self.log_dict(epoch_metrics)

        if self.wandb:
            self.wandb.log(epoch_metrics)

            # Create a seaborn heatmap from confusion matrix
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)
            sns.heatmap(conf_mat, annot=True, ax=ax, cmap="Blues", fmt="g", annot_kws={"fontsize":20, "fontweight": "bold"})
            # Make the font for the heatmap larger and bold
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight="bold")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")

            # Set the x ticks to be the self.labels
            ax.set_xticklabels(self.labels)
            ax.set_yticklabels(self.labels)

            # Make the x and y ticks bold and larger
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight="bold")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            self.wandb.log({"train_confusion_matrix_plot": wandb.Image(fig)})
            plt.close(fig)
            
        self.train_metric_monitor.reset()

    def on_validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]=None):
        """Validation epoch end hook.

        Arguments:
        ----------
        outputs (list):
            List of dictionaries containing the loss and the predictions.
        """
        
        current_epoch = self.current_epoch * 1.0
        epoch_metrics = self.valid_metric_monitor.compute()
                
        epoch_metrics["epoch"] = current_epoch

        conf_mat_key = list(filter(lambda x: "confusion_matrix" in x, epoch_metrics.keys()))
        conf_mat = epoch_metrics.pop(conf_mat_key[0])

        self.log_dict(epoch_metrics)

        if self.wandb:
            self.wandb.log(epoch_metrics)

            # Create a seaborn heatmap from confusion matrix
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            sns.heatmap(conf_mat, annot=True, ax=ax, cmap="Blues", fmt="g", annot_kws={"fontsize":20, "fontweight": "bold"})
            # Make the font for the heatmap larger and bold
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight="bold")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")

            # Set the x ticks to be the self.labels
            ax.set_xticklabels(self.labels)
            ax.set_yticklabels(self.labels)

            # Make the x and y ticks bold and larger
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight="bold")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            self.wandb.log({"valid_confusion_matrix_plot": wandb.Image(fig)})
            plt.close(fig)

        self.valid_metric_monitor.reset()


    def on_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]=None):
        """Test epoch end hook.

        Arguments:
        ----------
        outputs (list):
            List of dictionaries containing the loss and the predictions.
        """
        epoch_metrics = self.test_metric_monitor.compute()
        current_epoch = self.current_epoch
        epoch_metrics["epoch"] = current_epoch

        conf_mat_key = list(filter(lambda x: "confusion_matrix" in x, epoch_metrics.keys()))
        conf_mat = epoch_metrics.pop(conf_mat_key[0])
        self.log_dict(epoch_metrics)

        if self.wandb:
            self.wandb.log(epoch_metrics)

            # Create a seaborn heatmap from confusion matrix
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            sns.heatmap(conf_mat, annot=True, ax=ax, cmap="Blues", fmt="g", annot_kws={"fontsize":20, "fontweight": "bold"})
            # Make the font for the heatmap larger and bold
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight="bold")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")

            # Set the x ticks to be the self.labels
            ax.set_xticklabels(self.labels)
            ax.set_yticklabels(self.labels)

            # Make the x and y ticks bold and larger
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight="bold")
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight="bold")
            
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            self.wandb.log({"test_confusion_matrix_plot": wandb.Image(fig)})
            plt.close(fig)

        self.test_metric_monitor.reset()

    def create_loggers(
            self, num_classes: int, mutually_exclusive: bool, 
            label_names: List[str]):
        """Creates the loggers for the model."""

        multiclass_condition_1 = num_classes > 1 # Can not be binary
        
        if multiclass_condition_1:
            print(5*"\n")
            print("I am here ")
            print(5*"\n")
            assert len(label_names) == num_classes, \
                "The number of label names must be equal to the number of classes."
            
            task = "multiclass"
            num_classes = num_classes

            self.train_metric_monitor = MulticlassClassificationMetricMonitor(
                class_names=label_names,
                task = task,
                threshold = 0.5,
                num_classes = num_classes,
                prefix = "train"
            )

            self.valid_metric_monitor = MulticlassClassificationMetricMonitor(
                    class_names=label_names,
                    task = task,
                    threshold = 0.5,
                    num_classes = num_classes,
                    prefix = "valid"
                )

            self.test_metric_monitor = MulticlassClassificationMetricMonitor(
                    class_names=label_names,
                    task = task,
                    threshold = 0.5,
                    num_classes = num_classes,
                    prefix = "test"
                )
        else:
            task = "binary"
            num_classes = None

            self.train_metric_monitor = BinaryClassificationMetricMonitor(
                task = task,
                threshold = 0.5,
                prefix = "train"
            )
        
            self.valid_metric_monitor = BinaryClassificationMetricMonitor(
                task = task,
                threshold = 0.5,
                prefix = "valid"
            )

            self.test_metric_monitor = BinaryClassificationMetricMonitor(
                task = task,
                threshold = 0.5,
                prefix = "test"
            )

    def set_loss_and_activation(self, mutually_exclusive: bool, num_classes: int):
        """Sets the loss function.

        Arguments:
        ----------
        mutually_exclusive (bool):
            Whether the classes are mutually exclusive or not.
        """
        if mutually_exclusive:
            loss_weights = self.config["model"]["loss_weights"]
            loss_weights = torch.tensor(loss_weights)

            self.criterion = torch.nn.CrossEntropyLoss(loss_weights)
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
