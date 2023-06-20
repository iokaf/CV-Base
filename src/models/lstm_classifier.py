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


from src.models.base_classifier import BaseClassifier


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



class LSTMClassifier(BaseClassifier):
    """This class implements the RecurrentClassifier model."""

    def __init__(self, config: Dict, wandb: Any = None):
        """Initializes the model.
        
        Args:
            config (Dict): The configuration dictionary.
        """

        super().__init__(config=config, wandb=wandb)

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


