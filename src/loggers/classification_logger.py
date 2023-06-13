"""This module implements a class for calculating metrics for the classification
model.
"""
from typing import Dict, List
from torchmetrics import Accuracy, Precision, Recall, F1Score


class BinaryClassificationMetricMonitor:
    
        def __init__(
                self, 
                task = "binary",
                threshold = 0.5,
                prefix = "train", 
            ):
            
            self.prefix = prefix
    
            self.metrics = {
                f"{prefix}_accuracy": Accuracy(
                    task = task,
                    threshold=threshold,
                ),
    
                f"{prefix}_precision": Precision(
                    task = task,
                    threshold=threshold,
                ),
    
                f"{prefix}_recall": Recall(
                    task = task,
                    threshold=threshold,
                ),
    
                f"{prefix}_f1": F1Score(
                    task = task,
                    threshold=threshold,
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
            """
            predictions = step_resuts["predictions"].detach().cpu()

            # If we are predicting as two classes, get only the predictions for the
            # positive class.
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
            for metric in self.metrics.values():
                metric.update(
                    preds=predictions,
                    target=step_resuts["targets"].detach().cpu(),
                )
    
        def compute(self):
            """ Computes the values of all metrics and returns them in a dictionary.
    
            Returns:
            --------
                dict: Dictionary containing the values of all metrics.
            """
            results = {}
            for name, metric in self.metrics.items():
                results[name] = metric.compute()
            return results
    
        def reset(self):
            """ Resets all metrics to their initial state.
            """
            for metric in self.metrics.values():
                metric.reset()

class MulticlassClassificationMetricMonitor:

    def __init__(
            self,
            class_names: List[str],
            task = "multiclass",
            num_classes = None,
            threshold = 0.5,
            prefix = "train"
        ):
        
        self.prefix = prefix
        self.class_names = class_names

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

            f"{prefix}_per_class": Accuracy(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average=None,
            ),

            f"{prefix}_per_class_precision": Precision(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average=None,
            ),

            f"{prefix}_per_class_recall": Recall(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average=None,
            ),

            f"{prefix}_per_class_f1": F1Score(
                task = task,
                num_classes = num_classes,
                threshold=threshold,
                average=None,
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
            if "per_class" not in metric_name:
                metrics[metric_name] = metric.compute()
            else:

                # Calculate a list of the metric values for each class
                per_class_values = metric.compute()

                for class_idx, class_value in enumerate(per_class_values):

                    # Get class name
                    class_name = self.class_names[class_idx]

                    # Replace "per_class" with the class name to get the metric name
                    metric_name = metric_name.replace("per_class", class_name)

                    # Save the metric value
                    metrics[metric_name] = class_value
        
        return metrics
    
    def reset(self):
        """Resets the metric values."""
        for metric in self.metrics.values():
            metric.reset()
