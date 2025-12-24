import json
import os
from tqdm import tqdm
import shutil

def move_data(dataset_type='train'):
    """
    Move images into folders based on their categories.
    
    :param dataset_type: Type of dataset ('train' or 'val').
    """
    with open(f'./data/annotations/instances_{dataset_type}2017.json', 'r') as f:
        data = json.load(f)
        for i in data['categories']:
            os.makedirs(f'./data/{dataset_type}/' + i['name'], exist_ok=True)
        for i in tqdm(data['annotations'], desc=f'Moving {dataset_type} data'):
            img = [j['file_name'] for j in data['images'] if j['id'] == i['image_id']][0]
            cat = [j['name'] for j in data['categories'] if j['id'] == i['category_id']][0]
            if not os.path.exists(f'./data/{dataset_type}/' + cat + '/' + img):
                shutil.copy(f'./data/{dataset_type}/' + img, f'./data/{dataset_type}/' + cat + '/' + img)

if __name__ == '__main__':
    move_data('train')
    move_data('val')