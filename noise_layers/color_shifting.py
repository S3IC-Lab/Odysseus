import torch
import torch.nn as nn


class ColorShifting(nn.Module):
    """
    Randomly shifts the color of the image. The amount of shift is controlled by the parameters
    Brightness: brightness_range
    Saturation: saturation_range
    Hue: hue_range
    """
    def __init__(self, brightness_range, saturation_range, hue_range):
        super(ColorShifting, self).__init__()
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        brightness = torch.rand(1).item() * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
        saturation = torch.rand(1).item() * (self.saturation_range[1] - self.saturation_range[0]) + self.saturation_range[0]
        hue = torch.rand(1).item() * (self.hue_range[1] - self.hue_range[0]) + self.hue_range[0]

        noised_image = noised_image * saturation + brightness
        noised_image = noised_image + hue

        return [noised_image, noised_and_cover[1]]