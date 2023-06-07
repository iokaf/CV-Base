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
    ])

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
    augments = A.Compose([
        A.Resize(
            height=config["transforms"]["height"],
            width=config["transforms"]["width"],
        ),
        A.MotionBlur(p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        
        A.CLAHE(p=0.1),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.1),
        
        A.Flip(0.25),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
    ])

    return augments

