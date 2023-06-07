"""This module implements a class for calculating metrics for the classification
model.
"""
from typing import Dict
from torchmetrics import Accuracy, Precision, Recall, F1Score

class ClassificationMetricMonitor:

    def __init__(
            self, 
            task = "multiclass",
            num_classes = None,
            threshold = 0.5,
            prefix = "train", 
        ):
        
        self.prefix = prefix

        self.metrics = {
            f"{prefix}_micro_accuracy": Accuracy(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="micro",
            ),

            f"{prefix}_micro_precision": Precision(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="micro",
            ),

            f"{prefix}_micro_recall": Recall(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="micro",
            ),

            f"{prefix}_micro_f1": F1Score(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="micro",
            ),

            f"{prefix}_macro_accuracy": Accuracy(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="macro",
            ),

            f"{prefix}_macro_precision": Precision(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="macro",
            ),

            f"{prefix}_macro_recall": Recall(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="macro",
            ),

            f"{prefix}_macro_f1": F1Score(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average="macro",
            ),
        }

    def update(self, step_resuts: Dict):
        """ Updates all of the metric values with the given predictions and targets

        Arguments:
        ----------
        step_resuts (dict):
            Dictionary containing the predictions and the targets.
            It should contain the following:
            {
                "predictions": torch.Tensor,
                "targets": torch.Tensor,
            }

        Returns:
        --------
        dict: Dictionary containing the updated metric values.
        """
        preds = step_resuts["predictions"]
        targets = step_resuts["targets"]

        preds = preds.clone().detach().cpu()
        targets = targets.clone().detach().cpu()
        """Updates all of the metric values with the given predictions and targets"""
        metrics_step = {}
        for name, metric in self.metrics.items():
            value = metric(preds, targets.int())
            metrics_step[name] = value
        
        return metrics_step


    def compute(self):
        """Computes the metric values.
        
        Returns:
        --------
        dict: Dictionary containing the metric values.
        """
        metrics = {}
        for metric_name, metric in self.metrics.items():
            metrics[metric_name] = metric.compute()
        
        return metrics
    
    def reset(self):
        """Resets the metric values."""
        for metric in self.metrics.values():
            metric.reset()
