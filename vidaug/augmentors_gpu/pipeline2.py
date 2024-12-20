"""
Augmenters for pipeline 2 augmentations

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters_gpu as va`

List of augmenters:
    * SaltAndPepper
    * ElasticTransformation
    * GaussianBlur
"""
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ElasticTransform

class SaltAndPepper:
    """
    Augmenter to add salt(white) and pepper(black) noise.
    
    Args:
        salt_ratio (float): The ratio of salt noise. Default is 0.01.
        pepper_ratio (float): The ratio of pepper noise. Default is 0.01.
    """

    def __init__(self, salt_ratio=0.01, pepper_ratio=0.01):
        self.salt_ratio = salt_ratio
        self.pepper_ratio = pepper_ratio

    def __call__(self, clip):
        return [self._apply_salt_and_pepper(img) for img in clip]

    def _apply_salt_and_pepper(self, img):
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).cuda()
        
        salt_noise = torch.rand(img_tensor.shape[:2], device='cuda') < self.salt_ratio
        pepper_noise = torch.rand(img_tensor.shape[:2], device='cuda') < self.pepper_ratio
        
        img_tensor[:, salt_noise] = 255.0
        img_tensor[:, pepper_noise] = 0.0
        
        return F.to_pil_image(img_tensor.cpu().byte())

class ElasticTransformation:
    """
    Augmenter to apply elastic transformation.

    Args:
        alpha (float): The alpha parameter for elastic transformation. Default is 34.0.
        sigma (float): The sigma parameter for elastic transformation. Default is 4.0.
    """

    def __init__(self, alpha=34.0, sigma=4.0):
        self.elastic_transform = ElasticTransform(alpha=alpha, sigma=sigma)

    def __call__(self, clip):
        return [self.elastic_transform(img) for img in clip]

class GaussianBlur:
    """
    Augmenter to blur images using gaussian kernels.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, clip):
        return [self._apply_blur(img) for img in clip]

    def _apply_blur(self, img):
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).cuda()
        kernel_size = int(self.sigma * 6 + 1)  # Typically, kernel_size = 6*sigma + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel_size is odd
        blurred = F.gaussian_blur(img_tensor.unsqueeze(0), kernel_size, [self.sigma, self.sigma]).squeeze(0)
        return F.to_pil_image(blurred.cpu())