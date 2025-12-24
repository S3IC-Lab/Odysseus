import numpy as np
import torch.nn as nn
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.color_shifting import ColorShifting
from noise_layers.random_noise import RandomNoise
from noise_layers.identity import Identity


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device, n=1):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        self.n = n
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
            elif type(layer) is dict:
                if layer['type'] == 'crop':
                    self.noise_layers.append(Crop(layer['height_ratios'], layer['width_ratios']))
                elif layer['type'] == 'dropout':
                    self.noise_layers.append(Dropout(layer['keep_ratio_range']))
                elif layer['type'] == 'resize':
                    self.noise_layers.append(Resize(layer['resize_ratio_range']))
                elif layer['type'] == 'color_shifting':
                    self.noise_layers.append(ColorShifting(layer['brightness_range'], layer['saturation_range'], layer['hue_range']))
                elif layer['type'] == 'random_noise':
                    self.noise_layers.append(RandomNoise(layer['stddev_range']))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)

    def forward(self, encoded_and_cover):
        # randomly select one noise layer
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

