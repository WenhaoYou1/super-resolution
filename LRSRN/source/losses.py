import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import segmentation_models_pytorch as smp
import lpips


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss


class LPIPSLoss(nn.Module):
    def __init__(self, net='vgg', device='cuda'):
        super(LPIPSLoss, self).__init__()
        # Initialize LPIPS with chosen backbone ('vgg', 'alex', or 'squeeze')
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        # LPIPS models have requires_grad=False by default (frozen)
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # LPIPS expects inputs normalized to [-1, 1]
        # If your images are in [0, 1], scale them
        if pred.min() >= 0 and pred.max() <= 1:
            pred = pred * 2 - 1
            target = target * 2 - 1

        loss = self.loss_fn(pred, target)
        # LPIPS returns a tensor of shape [N, 1, 1, 1]; squeeze it
        return loss.mean()
        
# class VGG_gram(nn.Module):
#     def __init__(self):
#         super(VGG_gram, self).__init__()
#         vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
#         self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
#         for param in self.vgg16_conv_4_3.parameters():
#             param.requires_grad = False
    
#     def gram_matrix(self, x):
#         n, c, h, w = x.size()
#         x = x.view(n*c, h*w)
#         gram = torch.mm(x,x.t()) # 행렬간 곱셈 수행
#         return gram


#     def forward(self, output, gt):
#         vgg_output = self.vgg16_conv_4_3(output)
#         vgg_output = self.gram_matrix(vgg_output)

#         with torch.no_grad():
#             vgg_gt = self.vgg16_conv_4_3(gt.detach())
#             vgg_gt = self.gram_matrix(vgg_gt)
            
#         loss = F.mse_loss(vgg_output, vgg_gt)

#         return loss
    
def get_criterion(cfg, device):
    if cfg.loss == 'l1':
        return nn.L1Loss().cuda(device)
    elif cfg.loss == 'l2':
        return nn.MSELoss()
    elif cfg.loss == 'vgg':
        return VGG(device)
    elif cfg.loss == 'psnr_loss':
        return PSNRLoss().cuda(device)
    elif cfg.loss == 'lpips':
        return LPIPSLoss(net='vgg', device=device)
    else: 
        raise NameError('Choose proper model name!!!')

if __name__ == "__main__":
    # true = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # pred = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # loss = get_criterion(pred, true)
    # print(loss)
    #loss = nn.L1Loss()
    loss = PSNRLoss()
    
    predict = torch.tensor([1.0, 2, 3, 4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, 1, 1, 1], dtype=torch.float64,  requires_grad=True)
    mask = torch.tensor([0, 0, 0, 1], dtype=torch.float64, requires_grad=True)
    out = loss(predict, target, mask)
    out.backward()
    print(out)