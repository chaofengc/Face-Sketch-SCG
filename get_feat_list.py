from models.search_dataset import read_img_tensor
from models.loss import VGGFeat  
from data.image_folder import make_dataset

import torch
import os
from tqdm import tqdm

def get_featureset(photo_dir, file_list_path, save_feat_path):
    device = torch.device('cuda')
    vgg_model = VGGFeat(None, './pretrain_model/vgg_conv.pth').to(device)

    photoset = make_dataset(photo_dir)
    file_list = open(file_list_path, 'w+')

    feat_list = []
    for img_path in tqdm(photoset, total=len(photoset)):
        imgtensor = read_img_tensor(img_path, size=(224, 224)).to(device)
        tmp_feat = vgg_model(imgtensor, ['r51'])[0]
        feat_list.append(tmp_feat)

        file_list.write(f'./dataset/WildSketch/train_photos/{os.path.basename(img_path)}\n')
    
    torch.save(torch.cat(feat_list, dim=0), save_feat_path)

    file_list.close()

if __name__ == '__main__':
    get_featureset('./dataset/WildSketch/train_photos', './dataset/wildsketch_reference_img_list.txt', './dataset/wildsketch_feature_dataset.pth')