import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F
import Net
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
size = 448 if torch.cuda.is_available() else 128
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((size, size)),
    torchvision.transforms.ToTensor()
])
untrans = torchvision.transforms.ToPILImage()


def read_img(file):
    img=Image.open(file)
    img=trans(img).unsqueeze(0)
    img=img.detach().to(device)
    return img


def write_img(img, file):
    img = img.detach().cpu()
    img=img.squeeze(0)
    img=untrans(img)
    img.save(file)


c_img = read_img('test.jpg')
s_img = read_img('style.jpg')
input_img = c_img.clone().detach()
write_img(input_img, 'in.jpg')
optim = torch.optim.LBFGS([input_img.requires_grad_()])
model=Net.Model(c_img,s_img,device)
for i in range(20):
    def closure():
        # correct the values of updated input image
        input_img.data.clamp_(0, 1)
        optim.zero_grad()
        s_loss,c_loss=model(input_img)
        s_loss=s_loss*1000000
        loss=s_loss+c_loss
        loss.backward()
        if (i+1) % 3 == 0:
            print("it {}:".format(i))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format( s_loss.item(), c_loss.item()))
        return loss
    optim.step(closure)


input_img.data.clamp_(0, 1)
write_img(input_img,'out.jpg')