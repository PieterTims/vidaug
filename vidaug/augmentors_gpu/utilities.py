import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple

class Resize:
    """
    Augmenter to resize each frame in the clip to a specified dimension.

    Args:
        size (Tuple[int, int]): The target size (width, height) for resizing.
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, clip: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for img in clip:
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32).cuda()
            resized_img_tensor = F.resize(img_tensor, self.size)
            result.append(resized_img_tensor.cpu().numpy())
        return result

class Stretch:
    """
    Augmenter to increase the length of the clip by duplicating frames.

    Args:
        stretch_factor (int): The factor by which to stretch the clip.
    """

    def __init__(self, stretch_factor: int):
        self.stretch_factor = stretch_factor

    def __call__(self, clip: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for img in clip:
            for _ in range(self.stretch_factor):
                result.append(img)
        return result
