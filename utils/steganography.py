import os
import torch
import utils.model_utils as model_utils

from utils.options import *
from model.model import Stego
from noise_layers.noiser import Noiser
from torchvision import transforms
from PIL import Image
import math
import json

folder = json.load(open('./utils/config.json'))['stego-folder']
crop_size = 128

def string_to_binary_tensor(input_string, message_length, device):
    """
    Converts a string into a binary tensor of specified length.
    """
    binary_string = ''.join(format(ord(char), '08b') for char in input_string)
    
    binary_list = [int(bit) for bit in binary_string]
    
    if len(binary_list) < message_length:
        binary_list += [0] * (message_length - len(binary_list))
    elif len(binary_list) > message_length:
        binary_list = binary_list[:message_length]
    
    binary_tensor = torch.Tensor(binary_list).view(1, -1).to(device)
    
    return binary_tensor

def binary_tensor_to_string(binary_tensor):
    """ Converts a binary tensor back into a string. """
    binary_list = binary_tensor.view(-1).tolist()
    
    binary_string = ''.join([str(int(bit)) for bit in binary_list])

    string = ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, 32, 8))

    return string


def get_stego_image(text, image):
    """
    Hides the given text into the provided image using steganography.
    
    :param text: The text to hide.
    :param image: The PIL Image to hide the text in.
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    options_file = os.path.join(folder, 'options-and-config.pickle')
    _, hidden_config, noise_config = model_utils.load_options(options_file)
    checkpoint, _ = model_utils.load_last_checkpoint(os.path.join(folder, 'checkpoints'))

    noiser = Noiser(noise_config, device)
    model = Stego(hidden_config, device, noiser, None)
    model_utils.model_from_checkpoint(model, checkpoint)

    # text split into chunks of 4 characters
    texts = []
    for i in range(0, len(text), 4):
        chunk = text[i:i+4]
        if len(chunk) < 4:
            chunk += ' ' * (4 - len(chunk))
        texts.append(chunk)

    cat = image.convert('RGB').crop((
        (image.width - crop_size) // 2,
        (image.height - crop_size) // 2,
        (image.width + crop_size) // 2,
        (image.height + crop_size) // 2,
    ))

    width = crop_size * math.ceil(math.sqrt(len(texts)))
    new_image = Image.new('RGB', (width, width), (255, 255, 255))

    block_index = 0

    for x in range(0, width - crop_size + 1, crop_size):
        for y in range(0, width - crop_size + 1, crop_size):

            if block_index < len(texts):
                # steganography
                message = string_to_binary_tensor(
                    texts[block_index],
                    hidden_config.message_length,
                    device
                )

                block_tensor = transforms.ToTensor()(cat).unsqueeze(0).to(device)

                block = model.get_image(block_tensor, message)
                block = transforms.ToPILImage()(block.squeeze(0).cpu())

                new_image.paste(block, (x, y))
            block_index += 1

            if block_index >= len(texts):
                continue

    return new_image

def get_stego_text(image):
    """ 
    Extracts hidden text from the provided stego image. 

    :param image: The PIL Image to extract the text from.
    """
    image = image.convert('RGB')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    options_file = os.path.join(folder, 'options-and-config.pickle')
    _, hidden_config, noise_config = model_utils.load_options(options_file)
    checkpoint, _ = model_utils.load_last_checkpoint(os.path.join(folder, 'checkpoints'))

    noiser = Noiser(noise_config, device)
    model = Stego(hidden_config, device, noiser, None)
    model_utils.model_from_checkpoint(model, checkpoint)

    width, height = image.size
    res = ""

    # split image into blocks and decode each block
    for x in range(0, width - crop_size + 1, crop_size):
        for y in range(0, height - crop_size + 1, crop_size):
            block = image.crop((x, y, x + crop_size, y + crop_size))
            if all(pixel == (255, 255, 255) for pixel in block.getdata()):
                continue
            block_tensor = transforms.ToTensor()(block).unsqueeze(0).to(device)

            decoded_messages = model.get_message(block_tensor)
            text_segment = binary_tensor_to_string(decoded_messages)

            res += text_segment

    return res

def encode(message):
    binary_string = ''.join([format(ord(c), '08b') for c in message])
    return binary_string

def decode(binary_string):
    result = ''.join([chr(int(binary_string[i:i + 8], 2)) for i in range(0, len(binary_string), 8)])
    return result

def hide(img, index_image, message, path):
    """
    Hides the given message into the provided image.
    """
    pixels = img.load()
    width, height = img.size

    max_chars = (width * height) // 8
    message_length = len(message)
    if message_length > max_chars:
        message = message[:max_chars]

    bin_msg = format(message_length, '016b') + encode(message)
    total_bits = len(bin_msg)

    index = 0
    for i in range(width):
        for j in range(height):
            if index >= total_bits:
                break
            r, g, b = pixels[i, j]
            r = (r & 0xFE) | int(bin_msg[index])
            pixels[i, j] = (r, g, b)
            index += 1
    # img = get_stego_image(message, image=img)
    img.save(f"{path}/{index_image}.png")
    return f"{path}/{index_image}.png"

def extract(img):
    """
    Extracts hidden message from the provided image.
    """

    pixels = img.load()
    width, height = img.size

    length_bits = ''
    index = 0
    for i in range(width):
        for j in range(height):
            if index >= 16:
                break
            length_bits += str(pixels[i, j][0] & 1)
            index += 1

    message_length = int(length_bits, 2) * 8

    bin_msg = ''
    index = 0
    for i in range(width):
        for j in range(height):
            if index >= 16 + message_length:
                break
            bin_msg += str(pixels[i, j][0] & 1)
            index += 1
    # return get_stego_text(img)
    return decode(bin_msg[16:])

def getImages(args, EN):
    """
    Generates stego images for the dataset based on the provided arguments.
    
    :param args: Arguments containing dataset and cover image information.
    :param EN: Encoder object for encoding the text.
    """
    dataset = args.dataset
    if not os.path.exists(f"./data/{dataset}"):
        os.makedirs(f"./data/{dataset}")
        with open(f"./data/{dataset}.txt", "r") as f:
            data = f.readlines()
            for j, line in enumerate(data):
                question = line.strip()
                hide(Image.open(args.cover_image), j, EN.encode(question), f"./data/{dataset}")
        print(f"Images for {dataset} dataset have been generated successfully.")