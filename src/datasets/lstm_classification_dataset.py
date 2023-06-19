"""This module implements the dataset for using with an LSTM model for classification
tasks.
"""
import os
import json
import random
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
        if item["fold"] % 10 in valid_folds:
            valid_data.append(item)

        elif item["fold"] % 10 in test_folds:
            test_data.append(item)
        else:
            train_data.append(item)
    

    # FixMe: Specific for the current dataset
    valid_data = [item for item in valid_data if item["first_cut"]]
    test_data = [item for item in test_data if item["first_cut"]]

    print("")
    print(100 * "-")

    train_positives = sum([item["last_cut"] for item in train_data])
    train_negatives = len(train_data) - train_positives

    valid_positives = sum([item["last_cut"] for item in valid_data])
    valid_negatives = len(valid_data) - valid_positives

    test_positives = sum([item["last_cut"] for item in test_data])
    test_negatives = len(test_data) - test_positives
    
    print(f"Train positives: {train_positives}")
    print(f"Train negatives: {train_negatives}")

    print(f"Valid positives: {valid_positives}")
    print(f"Valid negatives: {valid_negatives}")

    print(f"Test positives: {test_positives}")
    print(f"Test negatives: {test_negatives}")

    print(100 * "-")
    print("")

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
        input_sequence_length: int,
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

        self.input_sequence_length = input_sequence_length

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

        first_frame = item["frame_num"]
        last_frame = item["cut_frame"]

        available_frame_numbers = list(range(first_frame, last_frame + 1, 5))

        # If the available frames are less than the input sequence length, we
        # duplicate the last frame to fill the gap.
        if len(available_frame_numbers) < self.input_sequence_length:
            missing_frames = self.input_sequence_length - len(available_frame_numbers)
            extra_frames = missing_frames * [available_frame_numbers[-1]]
            selected_frame_numbers = available_frame_numbers + extra_frames

        # If the available frames are more than the input sequence length, we
        # keep the first and last frames and randomly select the remaining ones.

        elif len(available_frame_numbers) > self.input_sequence_length:
            start = available_frame_numbers[0]
            end = available_frame_numbers[-1]

            selected_frame_numbers = (
                [start] 
                + random.sample(
                    available_frame_numbers[1:-1], 
                    self.input_sequence_length - 2
                    ) 
                + [end]
            )

        # If the available frames are equal to the input sequence length, we
        # keep all the frames.
        else:
            selected_frame_numbers = available_frame_numbers

        image_paths = [
            os.path.join(self.images_dir, video_filename, f"{frame_number:07d}.jpg")
            for frame_number in selected_frame_numbers
        ]

        images = {
            f"image{i+1:04d}": image_path 
            for i, image_path in enumerate(image_paths[1:])
        }

        images["image"] = image_paths[0]


        for key, value in images.items():
            images[key] = cv2.imread(value)
            images[key] = cv2.cvtColor(images[key], cv2.COLOR_BGR2RGB)
        
        crop = item.get("crop")
        if crop is not None:
            for key, value in images.items():
                images[key] = value[crop[0]:crop[1], crop[2]:crop[3]]

        if self.augmentations is not None:
            augmented_images = self.augmentations(**images)
            images = {key: augmented_images[key] for key in images.keys()}

        transformed_images = self.transforms(**images)
        images = {key: transformed_images[key] for key in images.keys()}

        image_keys = list(images.keys())
        image_keys.sort()

        images = [images[key] for key in image_keys]
        
        images = torch.stack(images, dim=0)

        label = numpy.array(item["last_cut"]).astype(int)
        # label = numpy.atleast_1d(label)
        # label = torch.tensor(label, dtype=torch.long)
        result = {
            "image": images,
            "label": label,
        }

        return result
    
def create_datasets(
        split_data: Dict, 
        images_dir: str, 
        input_sequence_length: int,
        transforms: Any, 
        augmentations: Any
    ) -> Tuple[Dataset, Dataset, Dataset]:
    """Creates the train, validation and test datasets.

    Arguments:
    ----------
        split_data (dict): A dictionary containing the train, validation and test sets.
        images_dir (str): The path to the directory containing the images.
        input_sequence_length (int): The length of the input sequence.
        transforms (Any): The transforms to be applied to the data.
        augmentations (Any): The augmentations to be applied to the data.

    Returns:
    --------
        tuple: A tuple containing the train, validation and test datasets.
    """

    train_dataset = ClassificationDataset(
        data=split_data["train_data"],
        images_dir=images_dir,
        input_sequence_length=input_sequence_length,
        transforms=transforms,
        augmentations=augmentations,
    )

    valid_dataset = ClassificationDataset(
        data=split_data["valid_data"],
        images_dir=images_dir,
        input_sequence_length=input_sequence_length,
        transforms=transforms,
        augmentations=None,
    )

    test_dataset = ClassificationDataset(
        data=split_data["test_data"],
        images_dir=images_dir,
        input_sequence_length=input_sequence_length,
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
        input_sequence_length=config["data"]["input_sequence_length"],
        transforms=transforms,
        augmentations=augmentations,
    )

    print(f"Train dataset size: {len(datasets['train_dataset'])}")
    print(f"Valid dataset size: {len(datasets['valid_dataset'])}")
    print(f"Test dataset size: {len(datasets['test_dataset'])}")

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
 
    

