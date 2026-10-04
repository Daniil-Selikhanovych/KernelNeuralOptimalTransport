##########################################################
## Standard Imports
##########################################################
import matplotlib.pyplot as plt
import numpy as np
import os, sys, random
import glob
from tqdm import tqdm
import argparse
import cv2

from PIL import Image

import json


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.fid_score import calculate_frechet_distance

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
FID_WEIGHTS_PATH = "../../fid_model/pt_inception-2015-12-05-6726825d.pth"

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True,
                 use_downloaded_weights=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3(use_downloaded_weights=use_downloaded_weights)
        else:
            inception = _inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    
def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`

    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    if version >= (0, 6):
        kwargs['init_weights'] = False

    return torchvision.models.inception_v3(*args, **kwargs)
    
def fid_inception_v3(use_downloaded_weights=False):
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              aux_logits=False,
                              pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    if use_downloaded_weights:
        # state_dict = torch.load(FID_WEIGHTS_PATH, map_location=None)
        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    else:
        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception

class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


##########################################################
## DL Imports
##########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torch import autograd
from torchvision.utils import save_image
from torch.autograd import Variable

##########################################################
from src.plotters import plot_noise_interp_unequal, plot_inv_noise_interp_unequal

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()   

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 9999
torch.manual_seed(SEED)
# path = '../../../Data/CelebA/archive/img_align_celeba/img_align_celeba/'
path = ""
output_path = './output/CelebA_bags_to_shoes_64x64/'
pretrain_path = './pretrained/CelebA_bags_to_shoes_64x64/'
inception_path = './Eval/utils/output/CelebA_bags_to_shoes_64x64/'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


T = 301 # Total epochs
init_iter = 30000 # initial iter
restore_model = 0 # Restart training from init_iter checkpoint

##########################################################
size = 64 # Size of each image, [size,size]
channels = 3 # Number of channels, [channels,size,size]

num_workers = 10 # Used in data loader
devices = [0]


## Training parameters
BATCH_SIZE = 64 # Batch size
K_G = 5
K_psi = 1
lam_go = 0

lr_G = 1e-4
lr_psi = 1e-4

beta1D = 0.0
beta1G = 0.0
beta2D = 0.9
beta2G = 0.9


parallel = True # Data parallelization, if multiple gpus are available
save_model = True # saves weights of G and psi if true
save_every = 5000 # save weights of G and psi
log_every = 100 # print on console
test_every = 1000 # save transport samples
test_inception_every = 5000 # compute FID stats
# test_inception_every = 1

num_inception_imgs = 50000 # number of images used to compute FID

sigma = 0.3 # noise standard deviation

##########################################################
## Prepare Data
##########################################################
transform = transforms.Compose([
            # transforms.CenterCrop(140),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5)),
        ])


path_to_train_male = "/trinity/home/daniil.selikhanovych/OptimalTransportModeling/data/bags/train_bags_128"
path_to_train_female = "/trinity/home/daniil.selikhanovych/OptimalTransportModeling/data/shoes/train_shoes_128"
path_to_test_male = "/trinity/home/daniil.selikhanovych/OptimalTransportModeling/data/bags/test_bags_128"
path_to_test_female = "/trinity/home/daniil.selikhanovych/OptimalTransportModeling/data/shoes/test_shoes_128"

train_dataA = datasets.ImageFolder(path_to_train_male, transform=transform)
train_dataB = datasets.ImageFolder(path_to_train_female, transform=transform)

test_data = datasets.ImageFolder(path_to_test_male, transform=transform)
test_data_b = datasets.ImageFolder(path_to_test_female, transform=transform)
print('Train dataA: ', len(train_dataA), 'Train dataB: ', len(train_dataB), 'Test data: ', len(test_data) )

train_loaderA = torch.utils.data.DataLoader(train_dataA, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last = True)
train_loaderB = torch.utils.data.DataLoader(train_dataB, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last = True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False, drop_last = False)
test_loader_b = torch.utils.data.DataLoader(test_data_b, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False, drop_last = False)

train_loader_iteratorA = iter(train_loaderA)
train_loader_iteratorB = iter(train_loaderB)

test_loader_iterator = iter(test_loader)
test_loader_iterator_b = iter(test_loader_b)

##########################################################
## Main Modules
##########################################################
def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)

def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv

class TransportMap(torch.nn.Module):
    def __init__(self, out_channels=channels, features=256):
        super().__init__()
        self.act = nn.ReLU()

        self.ip = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        ##########################################################
        self.down1 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down2 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down3 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down4 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        ##########################################################
        
        self.up1 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])
        self.up2 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])
        self.up3 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])
        self.up4 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])

        self.op = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()           
            )

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x):
        x = self.ip(x)

        x1 = self._compute_cond_module(self.down1, x)
        x2 = self._compute_cond_module(self.down2, x1)
        x3 = self._compute_cond_module(self.down3, x2)
        x4 = self._compute_cond_module(self.down4, x3)


        y3 = self._compute_cond_module(self.up1, x4)
        y3 = y3 + x3

        y2 = self._compute_cond_module(self.up2, y3)
        y2 = y2 + x2

        y1 = self._compute_cond_module(self.up3, y2)
        # print(f"x1.shape = {x1.shape}, y1.shape = {y1.shape}")
        y1 = y1 + x1

        y = self._compute_cond_module(self.up4, y1)
        y = y + x

        op = self.op(y)
        return op

print('='*64)
print('Ki Architecture: \n')
G = TransportMap().to(device)
summary(G,(channels,size,size))
print('='*64)

# sys.exit()

##########################################################
class Psi(torch.nn.Module):
    def __init__(self, in_channels = channels, out_channels=1, features=256):
        super().__init__()
        
        ##########################################################
        self.down1 = nn.ModuleList([
            conv3x3(in_channels, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down2 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down3 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down4 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.op = nn.Linear(in_features=features*4*4, out_features=1)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x


    def forward(self, x):
        x = self._compute_cond_module(self.down1, x)
        x = self._compute_cond_module(self.down2, x)
        x = self._compute_cond_module(self.down3, x)
        x = self._compute_cond_module(self.down4, x)

        x = x.view(x.shape[0],-1)
        op = self.op(x)
        return op

print('='*64)
print('Psi Architecture: \n')
psi = Psi().to(device)
summary(psi,(channels,size,size))
print('='*64)


dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx], use_downloaded_weights=False).to(device)
freeze(model); 
# sys.exit()

###########################################################
# Embeddings
Q = lambda x: x.detach() 
INV_TRANSFORM = lambda x: 0.5*x + 0.5

path_to_data = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/shoes_64_test.json"

print(f"target dataset = {path_to_data}")
with open(path_to_data, 'r') as fp:
    data_stats = json.load(fp)
    mu_data, sigma_data = data_stats['mu'], data_stats['sigma']


def LoadModel(model, name='OTM', path=pretrain_path):
    model.load_state_dict(torch.load(path+name+'.ckpt'))
    print('Model loaded from '+path+name+'.ckpt')
    return model


def RunInference(iteration, test_loader_iterator=test_loader_iterator, test_loader = test_loader):
    G = TransportMap().to(device)
    if parallel:
        G = nn.DataParallel(G, devices)
    G = LoadModel(G, 'otm_g_it_'+str(iteration))
    G.eval() 

    images = []
    for (X, _) in test_loader:
        X = X.to(device)
        # X = Degrade(X.to(device))
        G_X = INV_TRANSFORM(G(X))
        
        
        G_X = G_X.cpu().detach()
        
        break

        # images.append(G_X)
        
    path_to_save = output_path+f'test_translated_images_iteration_{iteration}.png'
    print(f"saving {path_to_save}")
    save_image(G_X.view(G_X.shape[0], channels, size, size), path_to_save, nrow=8, normalize=True)
    path_to_save = output_path+f'test_input_images_iteration_{iteration}.png'
    print(f"saving {path_to_save}")
    save_image(X.view(X.shape[0], channels, size, size), path_to_save, nrow=8, normalize=True)
    
    images = []
    
    pred_arr = []
    
    x_arr = []
    
    with torch.no_grad():
        for (X, _) in tqdm(test_loader):
            X = X.to(device)
            # X = Degrade(X.to(device))
            G_X = INV_TRANSFORM(G(X)).clamp(0., 1.)
            
            bs = X.shape[0]
            
            G_X = G_X.detach()
            
            pred_arr.append(G_X.permute((0, 2, 3, 1)))
            
            x_arr.append(INV_TRANSFORM(X).permute((0, 2, 3, 1)))
            
    pred_arr = torch.cat(pred_arr, axis=0)
    print(f"pred_arr.shape = {pred_arr.shape}")
    
    x_arr = torch.cat(x_arr, axis=0)
            
    return pred_arr, x_arr

iteration = 35000
pred_arr, x_arr = RunInference(iteration)

pred_arr = (pred_arr.cpu() * 255).numpy().astype(np.uint8)

path_to_save_preds = f"save_pred_bags_to_shoes_{iteration}"
os.makedirs(path_to_save_preds, exist_ok=True)

num_pred = pred_arr.shape[0]
for i in range(num_pred):
    cur_image = pred_arr[i]
    image = Image.fromarray(cur_image)
    
    img_name = f"test_{i}.png"
    path_to_save = os.path.join(path_to_save_preds, img_name)
    image.save(path_to_save)
    
# pred_arr, x_arr = RunInference(iteration)

x_arr = (x_arr.cpu() * 255).numpy().astype(np.uint8)

path_to_save_preds = f"input_bags_{iteration}"
os.makedirs(path_to_save_preds, exist_ok=True)

num_pred = x_arr.shape[0]
for i in range(num_pred):
    cur_image = x_arr[i]
    image = Image.fromarray(cur_image)
    
    img_name = f"test_{i}.png"
    path_to_save = os.path.join(path_to_save_preds, img_name)
    image.save(path_to_save)
    
    