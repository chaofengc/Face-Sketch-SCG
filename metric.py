import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import pyiqa
import torch
from torchvision import transforms


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_img(img_path):
    if not os.path.exists(img_path):
        if img_path.endswith('.png'):
            img_path = img_path.replace('.png', '.jpg')
        elif img_path.endswith('.jpg'):
            img_path = img_path.replace('.jpg', '.png')

    img = Image.open(img_path).convert('RGB')
    img = transforms.ToTensor()(img) 
    return img.unsqueeze(0)


def replace_name_if_not_exist(img_path):
    if not os.path.exists(img_path):
        if img_path.endswith('.png'):
            img_path = img_path.replace('.png', '.jpg')
        elif img_path.endswith('.jpg'):
            img_path = img_path.replace('.jpg', '.png')
    return img_path


def avg_dir_score(test_dir, gt_dir, metric_name='fsim', verbose=False):
    """
    Read images from two folders and calculate the average score.
    """
    metric_name = metric_name.lower()
    all_score = []
    if metric_name == 'fsim':
        model = pyiqa.create_metric('fsim')
    elif metric_name == 'lpips':
        model = pyiqa.create_metric('lpips-vgg')
    elif metric_name == 'dists':
        model = pyiqa.create_metric('dists')
    print('=========> Calculating {} score'.format(metric_name.upper()))

    for name in tqdm(sorted(os.listdir(gt_dir))):
        test_path = os.path.join(test_dir, name)
        gt_path = os.path.join(gt_dir, name)
        test_path = replace_name_if_not_exist(test_path)
        gt_path = replace_name_if_not_exist(gt_path)

        if not (is_image_file(test_path) or is_image_file(gt_path)):
            print(test_path, 'is not image')
            continue
        if not (os.path.exists(test_path) and os.path.exists(gt_path)): 
            print('Test or gt image missing:', name)
            continue
        test_img = read_img(test_path)
        gt_img = read_img(gt_path)
        tmp_score = model(test_img, gt_img).item()
        if verbose:
            print(f'Image: {name}, Metric: {metric_name}, Score: {tmp_score}')
        all_score.append(tmp_score)
    return np.mean(np.array(all_score))


def cal_all_scores(test_dir, gt_dir, method_name=None, direction='AtoB'):
    fsim_score = avg_dir_score(test_dir, gt_dir, 'fsim')
    lpips_score = avg_dir_score(test_dir, gt_dir, 'lpips')
    dists_score = avg_dir_score(test_dir, gt_dir, 'dists')

    if method_name is not None:
        print(f'{method_name}, {direction}, FSIM: {fsim_score:.4f}, LPIPS: {lpips_score:.4f}, DISTS: {dists_score:.4f}')
    elif direction is not None:
        print(f'{direction}, FSIM: {fsim_score:.4f}, LPIPS: {lpips_score:.4f}, DISTS: {dists_score:.4f}')
    else:
        print(f'FSIM: {fsim_score:.4f}, LPIPS: {lpips_score:.4f}, DISTS: {dists_score:.4f}')

    return fsim_score, lpips_score, dists_score
