"""This module contains the LSTMClassifier class.

This model takes a sequence of images and uses a neural network to extract the
features. 

Then, the features are passed to a LSTM neural network.

The last feature of the sequence is passed through a linear layer to get the
output.

The model is trained using the cross entropy loss function.
"""
import os
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import timm
import torch
import wandb

import matplotlib.pyplot as plt
import seaborn as sns

from src.loggers import BinaryClassificationMetricMonitor, MulticlassClassificationMetricMonitor


def get_positional_endoding(
    sequence_length: int, feature_size: int,
    device: torch.device) -> torch.Tensor:
    """Returns the positional encoding.

    FixMe: 
    This is a very naive implementation that should be replaced by a more
    efficient one.

    Arguments:
    ----------
    sequence_length (int):
        The length of the sequence.
    feature_size (int):
        The size of the features.
    device (torch.device):
        The device to use.

    Returns:
    --------
    torch.Tensor:
        The positional encoding. This is a tensor of shape
        (sequence_length, feature_size).
    """

    encoding = torch.zeros(sequence_length, feature_size, device=device)

    for pos in range(sequence_length):
        for i in range(0, feature_size, 2):

            sine_input = pos / (10000 ** ((2 * i) / feature_size))
            sine_input = torch.tensor(sine_input, device=device)

            cosine_input = pos / (10000 ** ((2 * (i + 1)) / feature_size))
            cosine_input = torch.tensor(cosine_input, device=device)

            encoding[pos, i] = torch.sin(sine_input)
            encoding[pos, i + 1] = torch.cos(cosine_input)
    
    return encoding



class LSTMClassifier(pl.LightningModule):
    """This class implements the RecurrentClassifier model."""

    def __init__(self, config: Dict, wandb: Any = None):
        """Initializes the model.
        
        Args:
            config (Dict): The configuration dictionary.
        """

        super().__init__()

        self.config = config

        self.use_scheduler = config["optimizer"]["use_scheduler"]
        self.labels = config["data"]["label_names"]
        self.wandb = wandb


        self.__feature_extractor = self.__create_feature_extractor(config)
        self.__num_features = self.__compute_features_shape(config)
        self.__lstm = self.__create_lstm(config)

        self.__classifier = torch.nn.Linear(
            config["model"]["lstm"]["hidden_size"],
            config["model"]["num_classes"]
        )


        if config["model"]["batch_norm"]:
            self.__batch_norm = torch.nn.BatchNorm1d(
                config["model"]["lstm"]["hidden_size"]
            )
        else:
            self.__batch_norm = None

        self.use_positional_encoding = config["model"]["positional_encoding"]

        self.set_loss_and_activation(
            config["model"]["mutually_exclusive"],
            config["model"]["num_classes"],    
        )

        self.create_loggers(
            num_classes=config["model"]["num_classes"],
            mutually_exclusive=config["model"]["mutually_exclusive"],
            label_names=config["data"]["label_names"]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Arguments:
        ----------
        x (torch.Tensor):
            The input tensor. This is a tensor of shape (batch_size, sequence_length,
            channels, height, width).

        Returns:
        --------
        torch.Tensor:
            The output tensor. This is a tensor of shape (batch_size, num_classes).
        """

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        x = x.reshape(batch_size * sequence_length, *x.shape[2:])

        features = self.__feature_extractor(x)

        features = features.reshape(batch_size, sequence_length, -1)

        if self.use_positional_encoding:
            encoding = get_positional_endoding(
                sequence_length=sequence_length,
                feature_size=features.shape[2],
                device=self.config["model"]["device"]
            )

            # Make batch size repetitions of the encoding
            encoding = encoding.repeat(batch_size, 1, 1)
            features = features + encoding

        # FixMe: BatchNorm?

        lstm_features, _ = self.__lstm(features)

        # Keep only the last output
        last_feature = lstm_features[:, -1, :]

        if self.__batch_norm is not None:
            last_feature = self.__batch_norm(last_feature)

        output = self.__classifier(last_feature)

        output = output.squeeze()
        return output

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
        y = y
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

        multiclass_condition_1 = num_classes > 2 # Can not be binary
        multiclass_condition_2 = num_classes == 2 and not mutually_exclusive

        if multiclass_condition_1 or multiclass_condition_2:

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
        if mutually_exclusive and num_classes > 1:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0]))
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

    def __create_feature_extractor(self, config: Dict):
            """Creates the feature extractor.

            Arguments:
            ----------
            config (Dict): 
                The configuration dictionary.
            
            Returns:
            --------
            torch.nn.Module:
                The feature extractor.
            """

            extractor = timm.create_model(
                config["model"]["feature_extractor"]["name"],
                pretrained=config["model"]["feature_extractor"]["pretrained"],
                num_classes=0 # This returns the features
            )

            extractor = extractor.to(config["model"]["device"])

            if self.config["model"]["feature_extractor"]["use_checkpoint"]:
                
                cpt_path = self.config["model"]["feature_extractor"]["checkpoint_path"]
                assert os.path.exists(cpt_path), f"Checkpoint path {cpt_path} does not exist."

                cpt = torch.load(cpt_path, map_location=self.device)
                extractor.load_state_dict(cpt["state_dict"])

            if config["model"]["feature_extractor"]["freeze"]:
                for param in extractor.parameters():
                    param.requires_grad = False
        
            return extractor

    def __compute_features_shape(self, config: Dict) -> int:
        """Number of features extracted from the feature extractor.
        
        Notes:
        ------
        We create a random tensor and pass it through the feature extractor to
        get the number of features.

        Returns:
        --------
        int:
            The number of features.
        """
        img_height = config["transforms"]["height"]
        img_width = config["transforms"]["width"]
        img_channels = config["data"]["channels"]

        input = torch.rand(1, img_channels, img_height, img_width)
        input = input.to(config["model"]["device"])
        features = self.__feature_extractor(input)

        return features.shape[1]

    def get_feature_extractor(self) -> torch.nn.Module:
        """Returns the feature extractor.
        
        Returns:
            torch.nn.Module:
                The feature extractor.
        """

        return self.__feature_extractor

    def get_number_of_features(self) -> int:
        """Returns the number of features extracted by the feature extractor.
        
        Returns:
            int:
                The number of features.
        """

        return self.__num_features         

    def __create_lstm(self, config: Dict) -> torch.nn.Module:
        """Creates the LSTM.

        Arguments:
        ----------
        config (Dict):
            The configuration dictionary.

        Returns:
        --------
        torch.nn.Module:
            The LSTM.
        """
        features_dim = self.get_number_of_features()

        lstm = torch.nn.LSTM(
            input_size=features_dim,
            hidden_size=config["model"]["lstm"]["hidden_size"],
            num_layers=config["model"]["lstm"]["num_layers"],
            dropout=config["model"]["lstm"]["dropout"],
            batch_first=True
        )

        lstm = lstm.to(config["model"]["device"])

        return lstm


