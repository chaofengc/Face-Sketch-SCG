import torch
import torch.nn as nn
import torch.nn.parameter as Param
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as tf
from einops import rearrange


def extract_patches(img, patch_size=(3, 3), stride=(1, 1)):
    """
    Divide img into overlapping patches with stride = 1
    img: (b, c, h, w)
    output patches: (b, nH*nW, c, patch_size)
    """
    assert type(patch_size) in [int, tuple], 'patch size should be int or tuple int'
    assert type(stride) in [int, tuple], 'stride size should be int or tuple int'
    if type(stride) is int:
        stride = (stride, stride)
    if type(patch_size) is int:
        patch_size = (patch_size, patch_size)
    patches = img.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1]) 
    patches = patches.contiguous().view(img.shape[0], img.shape[1], -1, patch_size[0], patch_size[1])
    patches = patches.transpose(1, 2)
    return patches


def feature_mrf_loss_func(x, y, vgg_model=None, layer=[], match_img_vgg=[], topk=1, patch_size=3):
    assert isinstance(match_img_vgg, list), 'Parameter match_img_vgg should be a list'
    mrf_crit = MRFLoss(patch_size=patch_size, topk=topk)
    loss = [0., 0]
    if len(layer) == 0 or layer[0] == 'r11' or layer[0] == 'r12':
        mrf_crit.patch_size = (5, 5)
        mrf_crit.filter_patch_stride = 4
    if len(layer) == 0:
        return mrf_crit(x, y)
    x_feat = vgg_model(x, layer)
    y_feat = vgg_model(y, layer)
    match_img_feat = [vgg_model(m, layer) for m in match_img_vgg]
    if len(match_img_vgg) == 0:
        for pred, gt in zip(x_feat, y_feat):
            tmp_loss = mrf_crit(pred, gt)
            loss[0] = loss[0] + tmp_loss[0] 
            loss[1] = loss[1] + tmp_loss[1] 
    elif len(match_img_vgg) == 1:
        for pred, gt, match0  in zip(x_feat, y_feat, match_img_feat[0]):
            tmp_loss = mrf_crit(pred, gt, [match0])
            loss[0] = loss[0] + tmp_loss[0] 
            loss[1] = loss[1] + tmp_loss[1] 
    elif len(match_img_vgg) == 2:
        for pred, gt, match0, match1 in zip(x_feat, y_feat, match_img_feat[0], match_img_feat[1]):
            tmp_loss = mrf_crit(pred, gt, [match0, match1])
            loss[0] = loss[0] + tmp_loss[0] 
            loss[1] = loss[1] + tmp_loss[1] 
    return loss


class MRFLoss(nn.Module):
    """
    Feature level patch matching loss.
    """
    def __init__(self, patch_size=(3, 3), filter_patch_stride=1, compare_stride=1, topk=1):
        super(MRFLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        self.compare_stride = compare_stride
        self.filter_patch_stride = filter_patch_stride
        self.topk = topk

    def best_topk_match(self, x1, x2):
        """
        Best topk match.
        x1: reference feature, (B, C, H, W)
        x2: topk candidate feature patches, (B*topk, nH*nW, c, patch_size, patch_size)
        """
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=2)

        x2 = rearrange(x2, '(b k) nhw c h w -> b (k nhw) c h w', k=self.topk)

        best_match_idx = []
        for i in range(x1.shape[0]):
            cosine_dist = F.conv2d(x1[[i]], x2[i], stride=self.compare_stride)
            best_match_idx.append(cosine_dist.max(dim=1)[1])
        
        best_match_idx = torch.cat(best_match_idx, dim=0)
        return F.one_hot(best_match_idx, num_classes=x2.shape[1]).float()

    def forward(self, pred_style, target_style, match=[]):
        """
        pred_style: feature of predicted image 
        target_style: target style feature
        match: images used to match pred_style with target style 

        switch(len(match)):
            case 0: matching is done between pred_style and target_style
            case 1: matching is done between match[0] and target style
            case 2: matching is done between match[0] and match[1]
        """
        assert isinstance(match, list), 'Parameter match should be a list'
        target_style_patches = extract_patches(target_style, self.patch_size, self.filter_patch_stride)
        pred_style_patches = extract_patches(pred_style, self.patch_size, self.compare_stride)

        target_style_patches = rearrange(target_style_patches, '(b topk) nhnw c h w -> b (topk nhnw) c h w', topk=self.topk)

        if len(match) == 0:
            best_math_idx = self.best_topk_match(pred_style, target_style_patches)
        elif len(match) == 1:
            best_math_idx = self.best_topk_match(match[0], target_style_patches)
        elif len(match) == 2:
            match_patches = extract_patches(match[1], self.patch_size, self.filter_patch_stride)
            best_math_idx = self.best_topk_match(match[0], match_patches)
        
        new_target_style_patches = torch.einsum('bhwn, bncpq->bhwcpq', best_math_idx, target_style_patches)
        new_target_style_patches = rearrange(new_target_style_patches, 'b h w c p q -> b (h w) c p q')
        
        def get_gm(x):
            x = torch.mean(x, (3, 4))
            b, hw, c = x.shape
            G = torch.bmm(x.transpose(1, 2), x)
            return G / (c*hw)
        
        return self.mse(pred_style_patches, new_target_style_patches), self.mse(get_gm(pred_style_patches), get_gm(new_target_style_patches))

