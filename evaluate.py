from metric import cal_all_scores
from utils import utils
import os
from PIL import Image
import torch
from torchvision.transforms import transforms


def read_img_tensor(img_path, size=None, mode='RGB'):
    img = Image.open(img_path).convert(mode)
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    img_tensor = transforms.ToTensor()(img)
    return img_tensor.unsqueeze(0) * 255


def evaluate(model, opt):
    data = opt.train_style
    if data == 'cufs':
        input_dir = './dataset/CUFS/test_photos'
        gt_dir = './dataset/CUFS/test_sketches'
    elif data == 'cufsf':
        input_dir = './dataset/CUFSF/test_photos'
        gt_dir = './dataset/CUFSF/test_sketches'
    elif data == 'wildsketch':
        input_dir = './dataset/WildSketch/test_photos'
        gt_dir = './dataset/WildSketch/test_sketches'
    
    # directory to save temporary results
    test_dir = f'./tmp_results/{opt.name}_{data}/eval_sketches_out'
    os.makedirs(test_dir, exist_ok=True)

    resize_shape = (256, 256)   # model input size

    for img_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(test_dir, img_name)
        save_path = save_path.replace('.jpg', '.png')
        inp = read_img_tensor(input_path, resize_shape)
        inpshape = read_img_tensor(input_path)
        img_size = (inpshape.shape[3], inpshape.shape[2])
        with torch.no_grad():
            out_sketch = model.netG_A(inp)
            out_sketch_img = utils.tensor_to_img(out_sketch, save_path, size=img_size, mode='L', v_range=[0, 255])
    sketch_out_scores = cal_all_scores(test_dir, gt_dir, direction='AtoB')

    input_dir, gt_dir = gt_dir, input_dir
        
    test_dir = f'./tmp_results/{opt.name}_{data}/eval_photos_out'
    os.makedirs(test_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(test_dir, img_name)
        save_path = save_path.replace('.jpg', '.png')
        inp = read_img_tensor(input_path, resize_shape, 'L')
        inpshape = read_img_tensor(input_path)
        img_size = (inpshape.shape[3], inpshape.shape[2])
        with torch.no_grad():
            out_sketch = model.netG_B(inp)
            out_sketch_img = utils.tensor_to_img(out_sketch, save_path, size=img_size, mode='RGB', v_range=[0, 255])
    photo_out_scores = cal_all_scores(test_dir, gt_dir, direction='BtoA')

    return sketch_out_scores, photo_out_scores


