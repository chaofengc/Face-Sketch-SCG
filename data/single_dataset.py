from data.base_dataset import BaseDataset, get_transform
from data.base_dataset import complex_imgaug
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from torchvision.transforms import transforms

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc 
        self.opt = opt
        self.transform = transforms.ToTensor()
 
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        if self.opt.direction == 'AtoB':
            A_img = Image.open(A_path).convert('RGB')
        elif self.opt.direction == 'BtoA':
            A_img = Image.open(A_path).convert('L')
        
        org_img = A_img.copy()
        org_img = self.transform(org_img) * 255
        
        if self.opt.Gin_size is not None:
            A_img = A_img.resize((self.opt.Gin_size, self.opt.Gin_size), Image.BICUBIC)

        A = self.transform(A_img) * 255
        return {'A_img': A, 'A_paths': A_path, 'org_img': org_img}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
