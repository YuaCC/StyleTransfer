import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F

'''
x[1,c,h,w]
'''
def gram_matrix(x):
    _, c, h, w = x.shape
    x = torch.reshape(x, (c, h * w))
    x = torch.matmul(x, x.t())
    x = x / (c * h * w)
    return x


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor, device):
        super().__init__()
        self.target = target.clone().detach().to(device)

    def forward(self, input):
        return F.mse_loss(input, self.target)


class StyleLoss(nn.Module):
    def __init__(self, target: torch.Tensor, device):
        super().__init__()
        target = target.clone().detach().to(device)
        self.target = gram_matrix(target)

    def forward(self, input):
        input = gram_matrix(input)
        return F.mse_loss(input, self.target)


class Normalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor,device):
        super().__init__()
        self.mean = mean.clone().detach().to(device)
        self.std = std.clone().detach().to(device)

    def forward(self, img):
        return (img - self.mean[None, :, None, None]) / self.std[None, :, None, None]


class Model(nn.Module):
    def __init__(self, content_img, style_img, device):
        super().__init__()
        cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        self.normalizer = Normalizer(cnn_normalization_mean, cnn_normalization_std,device)
        layers = []
        layer=nn.Sequential()
        for name,m in cnn.named_children():
            if isinstance(m,nn.ReLU):
                m=nn.ReLU(False)
            layer.add_module(name,m)
            if isinstance(m, nn.Conv2d):
                layers.append(layer.to(device))
                if len(layers)>=5:
                    break
                layer=nn.Sequential()
        self.layer1=layers[0]
        self.layer2=layers[1]
        self.layer3=layers[2]
        self.layer4=layers[3]
        self.layer5=layers[4]

        self.target_c=[]
        content_img=self.normalizer(content_img)
        for layer in layers:
            content_img=layer(content_img)
            self.target_c.append(content_img.detach())
        self.target_s=[]
        style_img=self.normalizer(style_img)
        for layer in layers:
            style_img=layer(style_img)
            style_gram=gram_matrix(style_img)
            self.target_s.append(style_gram.detach())

    def forward(self, input):
        fea=self.normalizer(input)
        fea=self.layer1(fea)
        fea_gram=gram_matrix(fea)
        style_loss=F.mse_loss(fea_gram,self.target_s[0])

        fea=self.layer2(fea)
        fea_gram=gram_matrix(fea)
        style_loss=style_loss+F.mse_loss(fea_gram,self.target_s[1])

        fea=self.layer3(fea)
        fea_gram=gram_matrix(fea)
        style_loss=style_loss+F.mse_loss(fea_gram,self.target_s[2])

        fea=self.layer4(fea)
        fea_gram=gram_matrix(fea)
        style_loss=style_loss+F.mse_loss(fea_gram,self.target_s[3])
        content_loss=F.mse_loss(fea,self.target_c[3])

        fea=self.layer5(fea)
        fea_gram=gram_matrix(fea)
        style_loss=style_loss+F.mse_loss(fea_gram,self.target_s[4])
        return style_loss,content_loss
