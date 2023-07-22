import torch
from torchvision import models
from utils import utils
from torch import nn, autograd
import torch.nn.functional as F

def tv_loss(x):
    """
    Total Variation Loss.
    """
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            ) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


class VGGFeat(nn.Module):
    """VGG19 model.
    ---------------------
    Codes borrowed from: https://github.com/leongatys/PytorchNeuralStyleTransfer
    """
    def __init__(self, opt, weight_path, pool_ks=2, pool_st=2):
        super().__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
        self.pool4 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
        self.pool5 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)

        # map_location = {'cuda:%d' % 0: 'cuda:%d' % opt.local_rank}
        map_location = 'cuda'
        self.load_state_dict(torch.load(weight_path, map_location=map_location))
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        # Input range [0, 255], RGB format
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            x = x[:, [2, 1, 0]] # Convert RGB to BGR
              
        mean = torch.Tensor([103.939, 116.779, 123.680]).to(x) # imagenet mean in BGR format
        x = x - mean.view(1, 3, 1, 1)
        return x 
        
    def forward(self, x, out_keys):
        x = self.preprocess(x)

        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class PCPLoss(torch.nn.Module):
    """Perceptual Loss.
    """
    def __init__(self, 
            opt, 
            layer=5,
            model='vgg',
            ):
        super(PCPLoss, self).__init__()

        self.mse = torch.nn.L1Loss()
        #  self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # self.weights = [1, 1, 1, 1, 1]

    def forward(self, x_feats, y_feats):
        loss = 0

        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.mse(xf, yf.detach())
        
        return loss 


class StyleLoss(torch.nn.Module):
    """Perceptual Loss.
    """
    def __init__(self,  ):
        super().__init__()

        self.mse = torch.nn.MSELoss()
        self.weights = [1, 1, 1, 1, 1]

    def get_mean_std(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat((mean, std), dim=1)

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf, w in zip(x_feats, y_feats, self.weights): 
            x_mean_std = self.get_mean_std(xf)
            y_mean_std = self.get_mean_std(yf)
            loss = loss + self.mse(x_mean_std, y_mean_std.detach()) 
        return loss 


class GMLoss(torch.nn.Module):
    """Perceptual Loss.
    """
    def __init__(self,  ):
        super().__init__()

        self.mse = torch.nn.MSELoss()
        self.weights = [1, 1, 1, 1, 1]

    def get_gm(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        G = torch.bmm(x, x.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf, w in zip(x_feats, y_feats, self.weights): 
            loss = loss + self.mse(self.get_gm(xf), self.get_gm(yf)) 
        return loss 


class FMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.mse(xf, yf.detach()) 
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'relative_hinge':
            pass
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode in ['softwgan']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        print('Using GAN loss:', gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    loss = nn.ReLU()(1 - prediction).mean()
                else:
                    loss = nn.ReLU()(1 + prediction).mean() 
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss  = - prediction.mean()
            return loss
        elif self.gan_mode == 'relative_hinge':
            real_preds, fake_preds = prediction, target_is_real 
            assert real_preds.shape == fake_preds.shape, "The inputs of relative hinge should be (real_preds, fake_preds)"
            real_fake_diff = real_preds - fake_preds.mean(dim=0, keepdim=True)
            fake_real_diff = fake_preds - real_preds.mean(dim=0, keepdim=True)
            if for_discriminator:
                loss = nn.ReLU()(1 - real_fake_diff).mean() + nn.ReLU()(1 + fake_real_diff).mean()
            else:
                loss = nn.ReLU()(1 + real_fake_diff).mean() + nn.ReLU()(1 - fake_real_diff).mean()
            return loss
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'softwgan':
            if target_is_real:
                loss = F.softplus(-prediction).mean()
            else:
                loss = F.softplus(prediction).mean()
        return loss



