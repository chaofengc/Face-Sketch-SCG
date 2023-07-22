import argparse
import os
import glob
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

from metric import cal_all_scores 
from utils import utils
from utils.download_util import load_file_from_url
from models.networks import ResnetGenerator, apply_norm


pretrain_model_urls = {
    'wildsketch_AtoB': 'https://github.com/chaofengc/Face-Sketch-SCG/releases/download/v0.1/wildsketch_net_G_A.pth',
    'cufs_AtoB': 'https://github.com/chaofengc/Face-Sketch-SCG/releases/download/v0.1/cufs_net_G_A.pth',
    'cufs_BtoA': 'https://github.com/chaofengc/Face-Sketch-SCG/releases/download/v0.1/cufs_net_G_B.pth',
    'cufsf_AtoB': 'https://github.com/chaofengc/Face-Sketch-SCG/releases/download/v0.1/cufsf_net_G_A.pth',
    'cufsf_BtoA': 'https://github.com/chaofengc/Face-Sketch-SCG/releases/download/v0.1/cufsf_net_G_B.pth',
}


def imread_to_tensor(path, size, mode='RGB'):
    img = Image.open(path).convert(mode)
    w, h = img.size
    img = img.resize(size, Image.BICUBIC)
    img_tensor = TF.to_tensor(img)
    return img_tensor.unsqueeze(0) * 255, [w, h]


@torch.no_grad()
def main():
    """Test demo.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--style', type=str, default='cufs', help='test sketch style: [cufs, cufsf, wildsketch]')
    parser.add_argument('-d', '--direction', type=str, default='AtoB', help='Output folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-g', '--ground_truth', type=str, default=None, help='path for ground truth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_size = (256, 256)   # Test with slightly larger resolution for better quality

    model_url_key = f'{args.style}_{args.direction}'
    if args.weight is None:
        if model_url_key in pretrain_model_urls:
            weight_path = load_file_from_url(pretrain_model_urls[model_url_key], model_dir='./pretrain_models/')
        else:
            raise ValueError(f'No pretrain model for style [{args.style}] and direction [{args.direction}]')
    else:
        weight_path = args.weight

    if args.direction == 'AtoB':
        input_mode, output_mode = 'RGB', 'L'
        model = ResnetGenerator(3, 1, norm_type='gn', relu_type='silu').to(device)
    elif args.direction == 'BtoA':
        input_mode, output_mode = 'L', 'RGB'
        model = ResnetGenerator(1, 3, norm_type='gn', relu_type='silu').to(device)

    apply_norm(model) 
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    output_dir = os.path.join(args.output, f'{args.style}_{args.direction}')
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Generate images with style [{args.style}] from [{args.direction}]: {img_name:>10}')

        img_tensor, org_shape = imread_to_tensor(path, test_size, mode=input_mode)
        output = model(img_tensor.to(device))

        save_path = os.path.join(output_dir, img_name)
        output_img = utils.tensor_to_img(output, save_path, size=org_shape, mode=output_mode, v_range=[0, 255])
        
        pbar.update(1)
    pbar.close()

    if args.ground_truth is not None:
        results = cal_all_scores(output_dir, args.ground_truth, direction=args.direction)
    
    
if __name__ == '__main__':
    main()
