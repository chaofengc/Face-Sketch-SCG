import torch
from torch import nn
import itertools

from .base_model import BaseModel
from . import networks
from models import loss
from models import search_dataset
from models.mrf_loss import feature_mrf_loss_func

from utils.image_pool import  ImagePool
from utils import utils 
from utils.download_util import load_file_from_url

VGG_URL = 'https://github.com/chaofengc/Face-Sketch-SCG/releases/download/v0.1/vgg_conv.pth'


def add_noise(x, noise_level=20):
    x = x + torch.randn_like(x) * noise_level
    return torch.clamp(x, 0, 255)


class SemiCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--train_style', type=str, default='cufs', help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--flayers', type=str, default='r31-r41-r51', help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--sigma', type=int, default=0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--patch_k', type=int, default=3, help='patch size for patch match')
        if is_train:
            parser.add_argument('--topk', type=int, default=5, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for gan loss')
            parser.add_argument('--lambda_mrf', type=float, default=1.0, help='weight for patch style loss')
            parser.add_argument('--lambda_sty', type=float, default=1.0, help='weight for style loss (gram matrix loss)')
            parser.add_argument('--lambda_pcp', type=float, default=1.0, help='weight for perceptual loss of reconstruction')
            
        parser.add_argument('--load_dir', type=str, default=None, help='checkpoint directory for reloading weight')
        parser.add_argument('--reload_iter', type=str, default=None, help='resume iter of checkpoint')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        opt.data_device = opt.device

        if opt.train_style == 'cufs':
            self.ref_style_dataset = ['CUHK_student', 'AR', 'XM2VTS']
            self.ref_feature       = './dataset/cufs_feature_dataset.pth'
            self.ref_img_list      = './dataset/cufs_reference_img_list.txt'
        elif opt.train_style == 'cufsf':
            self.ref_style_dataset = ['CUFSF']
            self.ref_feature       = './dataset/cufsf_feature_dataset.pth'
            self.ref_img_list      = './dataset/cufsf_reference_img_list.txt'
        elif opt.train_style == 'wildsketch':
            self.ref_style_dataset = ['WildSketch']
            self.ref_feature       = './dataset/wildsketch_feature_dataset.pth'
            self.ref_img_list      = './dataset/wildsketch_reference_img_list.txt'
        
        self.feature_dataset = torch.load(self.ref_feature).to(opt.device)

        self.feature_loss_layers = [x for x in opt.flayers.split('-')]

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'MRF', 'D_B', 'G_B', 'FeatB']

        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        self.visual_names = visual_names_A + visual_names_B  

        self.model_names = ['G_A', 'G_B']
        self.load_model_names = ['G_A', 'G_B']

        netnorm = 'spectral_norm'
        self.netG_A = networks.define_G(opt, 3, 1, use_norm=netnorm, relu_type=opt.act_type)
        self.netG_B = networks.define_G(opt, 1, 3, use_norm=netnorm, relu_type=opt.act_type)

        if self.isTrain:  # define discriminators
            self.Gin_size = opt.Gin_size
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            # self.load_model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            self.load_model_names = ['G_A', 'G_B']
            self.netD_A = networks.define_D(opt, in_channel=1, use_norm=netnorm) 
            self.netD_B = networks.define_D(opt, in_channel=3, use_norm=netnorm)
            self.vgg_model = loss.VGGFeat(opt, weight_path=load_file_from_url(VGG_URL, './pretrain_models/')).to(opt.device)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.data_device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionPix= nn.L1Loss()
            self.criterionCycle = nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            g_params = [{'params': self.netG_A.parameters(), 'lr': opt.lr / 2}, {'params': self.netG_B.parameters(), 'lr': opt.lr / 2}]
            self.optimizer_G = torch.optim.Adam(g_params, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr*2, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        if opt.load_dir is not None and opt.reload_iter is not None:
            self.load_networks(opt.reload_iter, opt.load_dir)

    def set_input(self, input, cur_iters=None):
        self.real_vgg_face = input['VF'].to(self.device)

    def forward(self):
        # find topk matched photo-sketch pair
        topk_sketch_img, topk_photo_img = search_dataset.find_photo_sketch_batch(
                            self.real_vgg_face, self.feature_dataset, self.ref_img_list,
                            self.vgg_model, dataset_filter=self.ref_style_dataset, topk=self.opt.topk, Gin_size=self.Gin_size)
        b, c, h, w = self.real_vgg_face.shape

        # get best matched photo-sketch pair for discriminator
        self.real_cuhk_face = topk_photo_img.view(b, -1, c, h, w)[:, 0]
        self.real_cuhk_sketch = topk_sketch_img.view(b, -1, c, h, w)[:, 0].mean(dim=1, keepdim=True)

        # ********** photo ---> sketch *************
        self.fake_vgg_sketch = self.netG_A(self.real_vgg_face)
        
        mrf_loss = feature_mrf_loss_func(
                                self.fake_vgg_sketch, topk_sketch_img, self.vgg_model,
                                self.feature_loss_layers, [self.real_vgg_face, topk_photo_img], topk=self.opt.topk, patch_size=self.opt.patch_k)

        # ********* sketch ---> photo *************
        # append one pair of reference set to real set for better benchmark performance
        self.real_vgg_face = torch.cat([self.real_vgg_face, self.real_cuhk_face[[0]]], dim=0)
        G_B_input_sketch = torch.cat([self.fake_vgg_sketch, self.real_cuhk_sketch[[0]]], dim=0)
        self.fake_vgg_sketch_wofootprint = add_noise(G_B_input_sketch, self.opt.sigma)

        # sketch to photo
        self.rec_vgg_face = self.netG_B(self.fake_vgg_sketch_wofootprint)

        # calculate vgg features for photos
        photo_feat_loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.real_vgg_face_feat = self.vgg_model(self.real_vgg_face, photo_feat_loss_layers)
        self.rec_vgg_face_feat = self.vgg_model(self.rec_vgg_face, photo_feat_loss_layers)

        self.real_A = self.real_vgg_face
        self.fake_B = self.fake_vgg_sketch
        self.rec_A = self.rec_vgg_face
        self.real_B = self.real_cuhk_sketch
        
        self.loss_MRF = mrf_loss[0] * self.opt.lambda_mrf + mrf_loss[1] * self.opt.lambda_sty

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_vgg_sketch)

        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_cuhk_sketch, fake_B)
        
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.rec_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_vgg_sketch), True, for_discriminator=False) * self.opt.lambda_g

        # GAN loss D_B(G_B(B))
        D_B_score = self.netD_B(self.rec_A, return_feat=False)
        self.loss_G_B = self.criterionGAN(D_B_score, True, for_discriminator=False) * self.opt.lambda_g

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_FeatB = self.criterionPCP(self.rec_vgg_face_feat, self.real_vgg_face_feat)
        self.loss_FeatB = self.loss_FeatB * self.opt.lambda_pcp

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_MRF + self.loss_FeatB 
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def get_current_visuals(self, size=224):
        vis_num = 4
        out = []
        out.append(utils.tensor_to_numpy(self.real_A))
        out.append(utils.tensor_to_numpy(self.fake_B.repeat(1, 3, 1, 1)))
        out.append(utils.tensor_to_numpy(self.fake_vgg_sketch_wofootprint.repeat(1, 3, 1, 1)))
        out.append(utils.tensor_to_numpy(self.rec_A))
        out.append(utils.tensor_to_numpy(self.real_B.repeat(1, 3, 1, 1)))
        out.append(utils.tensor_to_numpy(self.real_cuhk_face))
        out.append(utils.tensor_to_numpy(self.real_cuhk_sketch))

        return [utils.batch_numpy_to_image(x[:vis_num], size) for x in out]
