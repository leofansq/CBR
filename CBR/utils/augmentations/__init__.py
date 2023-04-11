import logging
import numpy as np 

from .augmentations import (
    RandomHorizontallyFlip,
    Compose,
)

aug_list = [RandomHorizontallyFlip]
aug_params = [[0.5]]

def get_composed_augmentations():
    augmentations = []
    for aug, aug_param in zip(aug_list, aug_params):
        if aug_param[0] > 0:
            augmentations.append(aug(*aug_param))

    return Compose(augmentations)
