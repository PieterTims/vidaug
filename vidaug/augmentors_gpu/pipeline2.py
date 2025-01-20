import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

class BoxBlur:
    """
    Augmenter to apply box blur to image.

    Args:
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation.
    """

    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, clip):
        result = []
        for img in clip:
            img_tensor = torch.from_numpy(np.array(img)).float().cuda()
            adjusted_img_tensor = F.gaussian_blur(img=img_tensor, kernel_size=self.kernel_size, sigma=self.sigma)
            result.append(adjusted_img_tensor.cpu().detach().numpy())
        return result

class AdjustShear:
    """
    Augmenter to apply shear transformation to image.

    Args:
        shear (float): Shear factor.
    """

    def __init__(self, shear):
        self.shear = shear

    def __call__(self, clip):
        result = []
        for img in clip:
            img_tensor = torch.from_numpy(np.array(img)).float().cuda()
            adjusted_img_tensor = F.affine(img=img_tensor, angle=0, translate=[0, 0], scale=1.0, shear=self.shear)
            result.append(adjusted_img_tensor.cpu().detach().numpy())
        return result

class AdjustRotation:
    """
    Augmenter to apply rotation to images.

    Args:
        angle (float): Rotation angle in degrees.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, clip):
        result = []
        for img in clip:
            img_tensor = torch.from_numpy(np.array(img)).float().cuda()
            adjusted_img_tensor = F.rotate(img=img_tensor, angle=self.angle)
            result.append(adjusted_img_tensor.cpu().detach().numpy())
        return result
