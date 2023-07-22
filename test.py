#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np

from models_guided import Generator_F2S, Generator_S2F
from utils import mask_generator
from utils import QueueMask

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=400, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()


### URSSR
opt.dataroot_A = '/home/ckc/UP-ShadowGAN/shadow_URSSR/shadow_test'
opt.dataroot_B = '/home/ckc/UP-ShadowGAN/shadow_URSSR/shadow_free'

opt.im_suf_A = '.png'
opt.im_suf_B = '.png'

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)

###### Definition of variables ######
# Networks
netG_A2B = Generator_S2F(opt.input_nc, opt.output_nc)
netG_B2A = Generator_F2S(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.to(device)
    netG_B2A.to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
img_transform = transforms.Compose([
    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
###################################
to_pil = transforms.ToPILImage()

###### Testing######
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')
if not os.path.exists('output/mask'):
    os.makedirs('output/mask')

##################################### A to B // shadow to shadow-free
gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

mask_queue = QueueMask(gt_list.__len__())

mask_non_shadow = Variable(Tensor(1, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False)

for idx, img_name in enumerate(gt_list):
    print('predicting: %d / %d' % (idx + 1, len(gt_list)))

    # Set model input
    img = Image.open(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)).convert('RGB')
    w, h = img.size

    img_var = (img_transform(img).unsqueeze(0)).to(device)

    # Generate output
    temp_B = netG_A2B(img_var)
    fake_B = 0.5*(temp_B.data + 1.0)
    mask_queue.insert(mask_generator(img_var, temp_B))
    fake_B = np.array(transforms.Resize((h, w))(to_pil(fake_B.data.squeeze(0).cpu())))
    Image.fromarray(fake_B).save('output/B/%s' % img_name + opt.im_suf_A)
    mask_last = mask_queue.last_item()
    print('Generated images %04d of %04d' % (idx+1, len(gt_list)))

##################################### B to A
gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_B) if f.endswith(opt.im_suf_B)]

for idx, img_name in enumerate(gt_list):
    print('predicting: %d / %d' % (idx + 1, len(gt_list)))

    # Set model input
    img = Image.open(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B)).convert('RGB')
    w, h = img.size

    img_var = (img_transform(img).unsqueeze(0)).to(device)

    mask = mask_queue.rand_item()
    mask_cpu = np.array(transforms.Resize((h, w))(to_pil(((mask.data + 1 ) * 0.5).squeeze(0).cpu())))

    temp_A = netG_B2A(img_var, mask)

    fake_A = 0.5*(temp_A.data + 1.0)

    fake_A = np.array(transforms.Resize((h, w))(to_pil(fake_A.data.squeeze(0).cpu())))

    # Save image files
    Image.fromarray(fake_A).save('output/A/%s' % img_name + opt.im_suf_B)
    Image.fromarray(mask_cpu).save('output/mask/%s' % img_name + opt.im_suf_B)
    print('Generated images %04d of %04d' % (idx + 1, len(gt_list)))
