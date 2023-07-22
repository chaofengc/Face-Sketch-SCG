from PIL import Image
import random

from PIL import Image

from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class FSVGGDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.Gin_size
        self.shuffle = True if opt.isTrain else False 

        if opt.train_style == 'cufs':
            face = './dataset/CUFS/train_photos'
        elif opt.train_style == 'cufsf':
            face = './dataset/CUFSF/train_photos'
        elif opt.train_style == 'wildsketch':
            face = './dataset/WildSketch/train_photos'
        
        self.vggface = make_dataset(opt.vggface) + make_dataset(face)
        if self.shuffle:
            random.shuffle(self.vggface)

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                ])
    
    def __len__(self,):
        return len(self.vggface)

    def __getitem__(self, idx):
        vggface_img = Image.open(self.vggface[idx]).convert('RGB')
        vggface_img = vggface_img.resize((self.img_size, self.img_size), Image.BICUBIC)
        vggface_tensor = self.to_tensor(vggface_img) * 255

        return {'VF': vggface_tensor}

