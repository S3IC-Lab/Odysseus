import torch.nn as nn
import numpy as np
import torch


class RandomNoise(nn.Module):
    """
    Randomly adds noise to the image. The noise is drawn from a normal distribution with mean 0 and std 1.
    """
    def __init__(self, noise_std):
        """
        :param noise_std: The standard deviation of the noise
        """
        super(RandomNoise, self).__init__()
        self.noise_std = noise_std


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noise = torch.randn_like(noised_image) * self.noise_std
        noised_image = noised_image + noise
        noised_and_cover[0] = noised_image
        return noised_and_cover