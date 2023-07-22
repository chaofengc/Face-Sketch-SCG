import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import cv2 as cv
from skimage import io
from PIL import Image
import os
import subprocess
import gzip


def img_to_tensor(img_path, device, size=None, mode='rgb'):
    """
    Read image from img_path, and convert to (C, H, W) tensor in range [-1, 1]
    """
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    if mode=='bgr':
        img = img[..., ::-1]
    if size:
        img = cv.resize(img, size, cv.INTER_CUBIC)
    img = img / 255 * 2 - 1 
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device) 
    return img_tensor.float()

def tensor_to_img(tensor, save_path=None, size=None, mode='RGB', v_range=[-1, 1]):
    """
    mode: RGB or L (gray image)
    Input: tensor with shape (C, H, W)
    Output: PIL Image
    """
    if isinstance(size, int):
        size = (size, size)
    img_array = tensor.squeeze().cpu().numpy()
    img_array = img_array.clip(0, 255)
    if mode == 'RGB':
        img_array = img_array.transpose(1, 2, 0)

    if size is not None:
        img_array = cv.resize(img_array, size, interpolation=cv.INTER_CUBIC)

    if len(v_range):
        img_array = (img_array - v_range[0]) / (v_range[1] - v_range[0]) * 255
        img_array = img_array.clip(0, 255)

    img_array = img_array.astype(np.uint8)
    if save_path:
        img = Image.fromarray(img_array, mode)
        img.save(save_path)

    return img_array

def tensor_to_numpy(tensor):
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.data.cpu().numpy()

def batch_numpy_to_image(array, size=None):
    """
    Input: numpy array (B, C, H, W) in [-1, 1]
    """
    if isinstance(size, int):
        size = (size, size)

    out_imgs = []
    #  array = np.clip((array + 1)/2 * 255, 0, 255) 
    array = np.clip(array, 0, 255) 
    array = np.transpose(array, (0, 2, 3, 1))
    for i in range(array.shape[0]):
        if size is not None:
            tmp_array = cv.resize(array[i], size, cv.INTER_CUBIC)
        else:
            tmp_array = array[i]
        out_imgs.append(tmp_array)
    return np.array(out_imgs)

def batch_tensor_to_img(tensor, size=None):
    """
    Input: (B, C, H, W) 
    Return: RGB image, [0, 255]
    """
    arrays = tensor_to_numpy(tensor)
    out_imgs = batch_numpy_to_image(arrays, size)
    return out_imgs 


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def get_gpu_memory_map():
    """Get the current gpu usage within visible cuda devices.

    Returns
    -------
    Memory Map: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    Device Ids: gpu ids sorted in descending order according to the available memory.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ]).decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = sorted([int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')])
    else: 
        visible_devices = range(len(gpu_memory))
    gpu_memory_map = dict(zip(range(len(visible_devices)), gpu_memory[visible_devices]))
    return gpu_memory_map, sorted(gpu_memory_map, key=gpu_memory_map.get)


def save(obj, save_path):
    for k in obj.keys():
        obj[k] = obj[k].to('cpu')
    with gzip.GzipFile(save_path, 'wb') as f:
        torch.save(obj, f)

def load(read_path):
    if read_path.endswith('.gzip'):
        with gzip.open(read_path, 'rb') as f:
            weight = torch.load(f)
    else:
        weight = torch.load(read_path)
    return weight


if __name__ == '__main__':
    hm = torch.randn(32, 68, 128, 128).cuda()
    flip(hm, 2)
    x = torch.ones(32, 68)
    y = torch.ones(32, 68)
    print(get_gpu_memory_map())



