import os
import itertools
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def read_img_tensor(img_path, color=1, size=None):
    """
    Read image and convert it to Tensor in 0~255.
    Args:
        img_path: str, test image path
        size: tuple, output size (1, C, W, H)
    """
    if not os.path.exists(img_path):
        if img_path.endswith('.png'):
            img_path = img_path.replace('.png', '.jpg')
        elif img_path.endswith('.jpg'):
            img_path = img_path.replace('.jpg', '.png')

    if color:
        img = Image.open(img_path).convert('RGB') 
    else:
        img = Image.open(img_path).convert('L') 
    
    if size is not None:
        img = transforms.functional.resize(img, size)
    return transforms.functional.to_tensor(img).unsqueeze(0) * 255  


@torch.no_grad()
def find_photo_sketch_batch(photo_batch, dataset_feature, img_name_list, vgg_model, 
        topk=1, dataset_filter=['CUHK_student', 'AR'], compare_layer=['r51'], Gin_size=None):
    """
    Search the dataset to find the topk matching image.
    """
    dataset_all = dataset_feature
    img_name_list_all = np.array([x.strip() for x in open(img_name_list).readlines()])
    img_name_list     = []
    dataset_idx       = []
    for idx, i in enumerate(img_name_list_all):
        for j in dataset_filter:
            if j in i:
                img_name_list.append(i)
                dataset_idx.append(idx)
                break
    dataset = dataset_all[dataset_idx]
    img_name_list = np.array(img_name_list)

    photo_feat = vgg_model(photo_batch, compare_layer)[0]
    photo_feat = torch.nn.functional.normalize(photo_feat, p=2, dim=1)
    dataset    = torch.nn.functional.normalize(dataset, p=2, dim=1)

    dist = torch.einsum('ichw,nchw->in', photo_feat, dataset)
    topk_idx = torch.topk(dist, topk, 1)[1].cpu().tolist()
    img_idx = list(itertools.chain(*topk_idx))
    
    match_img_list    = img_name_list[img_idx]
    match_sketch_list = [x.replace('train_photos', 'train_sketches') for x in match_img_list]

    match_img_batch    = [read_img_tensor(x, size=(Gin_size, Gin_size)) for x in match_img_list]
    match_sketch_batch = [read_img_tensor(x, size=(Gin_size, Gin_size)) for x in match_sketch_list]
    match_sketch_batch, match_img_batch = torch.stack(match_sketch_batch).squeeze(), torch.stack(match_img_batch).squeeze()

    return match_sketch_batch.to(photo_batch), match_img_batch.to(photo_batch)
