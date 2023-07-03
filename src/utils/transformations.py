"""Define the transformations and augmentations used in the project."""
from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformations(config: Dict):
    """Define the transformations used in the project.
    
    Arguments:
    ----------
    config: Dict
        The configuration dictionary.

    Returns:
    --------
    transforms: albumentations.Compose
        The transformations to be applied to the images.
    """

    if "input_sequence_length" in config["data"]:
        # We have a -1 here because the first is there by default
        additional_targets = {
            f"image{i:04d}": "image" 
            for i in range(config["data"]["input_sequence_length"])
        }
    else:
        additional_targets = None


    transforms = A.Compose([
        A.Resize(
            height=config["transforms"]["height"],
            width=config["transforms"]["width"],
        ),
        A.Normalize(
            mean=config["transforms"]["mean"],
            std=config["transforms"]["std"],
            max_pixel_value=255.0,
            always_apply=True,
        ),
        ToTensorV2()
        ],
    additional_targets=additional_targets
    )

    return transforms
        
def get_augmentations(config: Dict):
    """Define the augmentations used in the project.
    
    Arguments:
    ----------
    config: Dict
        The configuration dictionary.

    Returns:
    --------
    augments: albumentations.Compose
        The augmentations to be applied to the images.
    """
    if "input_sequence_length" in config["data"]:
        # We have a -1 here because the first is there by default
        additional_targets = {
            f"image{i:04d}": "image" 
            for i in range(config["data"]["input_sequence_length"])
        }
    else:
        additional_targets = None

    augments = A.Compose([
        A.Resize(
            height=config["transforms"]["height"],
            width=config["transforms"]["width"],
        ),
        A.MotionBlur(p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        
        A.CLAHE(p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        
        A.GaussNoise(p=0.5),
        A.ISONoise(p=0.5),
        A.MultiplicativeNoise(p=0.5),

        A.CoarseDropout(
            min_holes=4,
            max_holes=8, 
            min_height=0.05,
            max_height=0.1,
            min_width=0.05,
            max_width=0.1, 
            p=0.5),
        

        A.Flip(0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ], 
        additional_targets=additional_targets
    )

    return augments

