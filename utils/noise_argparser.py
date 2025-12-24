import argparse
import re
from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.color_shifting import ColorShifting
from noise_layers.random_noise import RandomNoise


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    # example: crop((0.8,1.0),(0.8,1.0))
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    # example: dropout(0.1,0.3)
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    print(matches)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    # example: resize(0.8,1.0)
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))

def parse_color(color_command):
    # Brightness: brightness_range
    # Saturation: saturation_range
    # Hue: hue_range
    # example: color(0.01,0.03,0.01)
    matches = re.match(r'color\((\d+\.*\d*,\d+\.*\d*,\d+\.*\d*)\)', color_command)
    color_args = matches.groups()[0].split(',')
    brightness_range = (-float(color_args[0]), float(color_args[0]))
    saturation_range = (1 - float(color_args[1]), 1 + float(color_args[1]))
    hue_range = (-float(color_args[2]), float(color_args[2]))
    return ColorShifting(brightness_range, saturation_range, hue_range)

def parse_noise(noise_command):
    # stddev: the standard deviation of the noise
    # example: noise(0.1)
    matches = re.match(r'noise\((\d+\.*\d*)\)', noise_command)
    stddev = float(matches.groups()[0])
    return RandomNoise(stddev)

class NoiseArgParser(argparse.Action):
    """
    Custom argparse action to parse noise layer specifications from command line.
    """
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('crop')] == 'crop':
                layers.append(parse_crop(command))
            elif command[:len('dropout')] == 'dropout':
                layers.append(parse_dropout(command))
            elif command[:len('resize')] == 'resize':
                layers.append(parse_resize(command))
            elif command[:len('color')] == 'color':
                layers.append(parse_color(command))
            elif command[:len('noise')] == 'noise':
                layers.append(parse_noise(command))
            elif command[:len('jpeg')] == 'jpeg':
                layers.append('JpegPlaceholder')
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)
