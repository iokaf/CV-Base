"""This module implements the basic dataset class for classification tasks."""
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader


def load_data(json_path: str) -> List[Dict[str, Any]]:
    """Loads the data from a json file.

    Arguments:
    ----------
        json_path (str): Path to the json file.

    Returns:
    --------
        list: List of dictionaries containing the data.
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    return data

def train_valid_test_split(
    data: List[Dict[str, Any]],
    valid_folds: List[int],
    test_folds: List[int],
    ) -> Dict[str, Any]:
    """Splits the data into train, validation and test sets.

    Arguments:
    ----------
    data (list):
        List of dictionaries containing the data.

    valid_folds (list):
        List of integers containing the folds to be used for validation.
    
    test_folds (list):
        List of integers containing the folds to be used for testing.

    Returns:
    --------
        dict: A dictionary containing the train, validation and test sets.
    """

    train_data = []
    valid_data = []
    test_data = []

    for item in data:
        if item["fold"] in valid_folds:
            valid_data.append(item)
        elif item["fold"] in test_folds:
            test_data.append(item)
        else:
            train_data.append(item)
    
    return {
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
    }

class ClassificationDataset(Dataset):
    """The dataset class"""

    def __init__(
        self, 
        data: List[Dict[str, Any]],
        images_dir: str,
        transforms: Any,
        augmentations: Optional[Any] = None,
        ):
        """Initializes the dataset.
        
        Arguments:
        ----------
        data (list): 
            List of dictionaries containing the data.
        transforms (Any): 
            The transforms to be applied to the data.
        augmentations (Any): 
            The augmentations to be applied to the data.
        """

        self.data = data
        self.transforms = transforms
        self.augmentations = augmentations
        self.images_dir = images_dir

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
        --------
            int: Length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single item from the dataset.

        Notes:
        ------
        We ensure that labels is at least 2D, so that it can be used with
        torch.nn.CrossEntropyLoss.

        Arguments:
        ----------
            idx (int): Index of the item to be returned.

        Returns:
        --------
            dict: A dictionary containing the data.
        """
        item = self.data[idx]

        video_filename = item["video_filename"]

        if not video_filename.endswith(".mkv"):
            video_filename += ".mkv"

        frame_number = item["frame_num"]

        image_path = os.path.join(
            self.images_dir, video_filename, f"{frame_number:07d}.jpg"
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        crop = item.get("crop")
        if crop is not None:
            image = image[crop[0]:crop[1], crop[2]:crop[3]]

        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]

        image = self.transforms(image=image)["image"]
        
        label = numpy.array(item["last_cut"]).astype(int)
        # label = numpy.atleast_1d(label)
        # label = torch.tensor(label, dtype=torch.long)
        result = {
            "image": image,
            "label": label,
        }

        return result
    
def create_datasets(
        split_data: Dict, 
        images_dir: str, 
        transforms: Any, 
        augmentations: Any
    ) -> Tuple[Dataset, Dataset, Dataset]:
    """Creates the train, validation and test datasets.

    Arguments:
    ----------
        split_data (dict): A dictionary containing the train, validation and test sets.
        images_dir (str): The path to the directory containing the images.
        transforms (Any): The transforms to be applied to the data.
        augmentations (Any): The augmentations to be applied to the data.

    Returns:
    --------
        tuple: A tuple containing the train, validation and test datasets.
    """

    train_dataset = ClassificationDataset(
        data=split_data["train_data"],
        images_dir=images_dir,
        transforms=transforms,
        augmentations=augmentations,
    )

    valid_dataset = ClassificationDataset(
        data=split_data["valid_data"],
        images_dir=images_dir,
        transforms=transforms,
        augmentations=None,
    )

    test_dataset = ClassificationDataset(
        data=split_data["test_data"],
        images_dir=images_dir,
        transforms=transforms,
        augmentations=None,
    )

    return {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
    }


def get_dataloaders(
        config: Dict[str, Any],
        transforms: Any,
        augmentations: Any,
    ) -> Dict[str, DataLoader]:
    """Creates the dataloaders.

    Arguments:
    ----------
        data_path (str): The path to annotations file.
        config (dict): A dictionary containing the configuration parameters.

    Returns:
    --------
        dict: A dictionary containing the train, validation and test dataloaders.
    """
    data_path = config["data"]["annotations_path"]
    data = load_data(data_path)
    
    valid_folds = config["data"]["validation_folds"]
    test_folds = config["data"]["test_folds"]

    split_data = train_valid_test_split(
        data=data,
        valid_folds=valid_folds,
        test_folds=test_folds,
    )

    datasets = create_datasets(
        split_data=split_data,
        images_dir=config["data"]["images_directory"],
        transforms=transforms,
        augmentations=augmentations,
    )

    train_dataloader = DataLoader(
        dataset=datasets["train_dataset"],
        batch_size=config["dataloading"]["train_batch_size"],
        shuffle=True,
        num_workers=config["dataloading"]["train_workers"],
        pin_memory=config["dataloading"]["pin_memory"],
    )

    valid_dataloader = DataLoader(
        dataset=datasets["valid_dataset"],
        batch_size=config["dataloading"]["valid_batch_size"],
        shuffle=False,
        num_workers=config["dataloading"]["valid_workers"],
        pin_memory=config["dataloading"]["pin_memory"],
    )

    if len(datasets["test_dataset"]) > 0:
        test_dataloader = DataLoader(
            dataset=datasets["test_dataset"],
            batch_size=config["dataloading"]["test_batch_size"],
            shuffle=False,
            num_workers=config["dataloading"]["test_workers"],
            pin_memory=config["dataloading"]["pin_memory"],
        )
    else:
        test_dataloader = None

    return {
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
    }


class MyDataModule(pl.LightningDataModule):
    """Define a DataModule for your data.
    """

    def __init__(
            self, config: Dict[str, Any], transforms: Any,augmentations: Any):
        loaders = get_dataloaders(
            config=config,
            transforms=transforms,
            augmentations=augmentations,
        )

        self.train_dl = loaders["train_dataloader"]
        self.valid_dl = loaders["valid_dataloader"]
        self.test_dal = loaders["test_dataloader"]
        
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.valid_dl
    
    def test_dataloader(self):
        return self.test_dl
    

