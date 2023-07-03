"""This module contains the definition of an image classifier model"""

from typing import Any, Dict

import timm
import torch

from src.models.base_classifier import BaseClassifier

class Classifier(BaseClassifier):
    """Implementation of the classifier class."""

    def __init__(self, config: Dict, wandb: Any = None):
        """Initializes the classifier.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config=config, wandb=wandb)
        
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

        # Freeze any selected layers
        for name, parameter in self.model.named_parameters():
            for layer in config["model"]["freeze_layers"]:
                if layer in name:
                    parameter.requires_grad = False
                    break

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