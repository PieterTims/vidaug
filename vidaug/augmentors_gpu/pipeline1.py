"""
Augmenters for pipeline 1 augmentations

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters_gpu as va`

List of augmenters:
    * AdjustBrightness
    * AdjustContrast
    * AdjustSharpness
"""
import torch
from torchvision.transforms import functional as F
import numpy as np

class AdjustBrightness:
    """
    Augmenter to adjust brightness.

    Args:
        factor (float): Brightness adjustment factor.
            brightness_factor = 0, black output image.
            brightness_factor = 1, the original image.
            0 < brightness_factor < 1, a darker output image.
            brightness_factor > 1, a brighter output image.
    """
    
    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, clip):
        result = []
        for img in clip:
            img_tensor = torch.from_numpy(np.array(img)).float().cuda()
            adjusted_img_tensor = F.adjust_brightness(img_tensor, self.factor)
            result.append(adjusted_img_tensor.cpu().detach().numpy())
        return result

class AdjustContrast(object):
    """
    Augmenter to adjust contrast.

    Args:
        factor (float): Contrast adjustment factor.
            Solid gray image if contrast_factor = 0.
            The original image if contrast_factor = 1,
            To lower contrast use 0 < contrast_factor < 1.
            To increase contrast use contrast_factor > 1.
    """

    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, clip):
        result = []
        for img in clip:
            img_tensor = torch.from_numpy(np.array(img)).float().cuda()
            adjusted_img_tensor = F.adjust_contrast(img_tensor, self.factor)
            result.append(adjusted_img_tensor.cpu().detach().numpy())
        return result

class AdjustSharpness(object):
    """
    Augmenter to adjust sharpness.

    Args:
        factor (float): Sharpness adjustment factor.
            Blurred image if sharpness_factor = 0.
            The original image if sharpness_factor = 1,
            To lower sharpness use 0 < sharpness_factor < 1.
            To increase sharpness use sharpness_factor > 1.
    """

    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, clip):
        result = []
        for img in clip:
            img_tensor = torch.from_numpy(np.array(img)).float().cuda()
            adjusted_img_tensor = F.adjust_sharpness(img_tensor, self.factor)
            result.append(adjusted_img_tensor.cpu().detach().numpy())
        return result
