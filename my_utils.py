import h5py
import numpy as np
from tqdm import tqdm
import gc

import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, \
    RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
import torch.nn as nn

from inception import InceptionV3


def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')

    return TensorDataset(dataset, torch.zeros(len(dataset)))


def load_dataset(name, path, img_size=64, batch_size=64, test_ratio=0.1):
    if name in ['shoes', 'handbag', 'outdoor', 'church']:
        dataset = h5py_to_dataset(path, img_size)
    elif name in ['celeba_female', 'celeba_male', 'aligned_anime_faces']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['dtd']:
        transform = Compose(
            [Resize(300), RandomResizedCrop((img_size,img_size), scale=(128./300, 1.), ratio=(1., 1.)),
             RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
             ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = ImageFolder(path, transform=transform)

    if name in ['celeba_female', 'celeba_male']:
        with open('datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        if name == 'celeba_female':
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
    else:
        idx = list(range(len(dataset)))

    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_dataloader = DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size)
    return train_dataloader, test_dataloader


class Sampler:
    def __init__(
            self, device='cuda',
    ):
        self.device = device

    def sample(self, size=5):
        pass


class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)


def tensor2img(tensor):
    return tensor.to('cpu').permute(0, 2, 3, 1).mul(0.5).add(0.5).numpy().clip(0, 1)


def multiple_gpu_gen(gen, devices_id):
    gen.enc_style = nn.DataParallel(gen.enc_style, device_ids=devices_id)
    gen.enc_content = nn.DataParallel(gen.enc_content, device_ids=devices_id)
    gen.dec = nn.DataParallel(gen.dec, device_ids=devices_id)
    gen.mlp = nn.DataParallel(gen.mlp, device_ids=devices_id)
    return gen


def multiple_gpu_dis(dis, devices_id):
    for i in range(len(dis.cnns)):
        dis.cnns[i] = nn.DataParallel(dis.cnns[i], device_ids=devices_id)

    dis.downsample = nn.DataParallel(dis.downsample, device_ids=devices_id)
    return dis


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def make_prediction(x_a, s_b2, trainer):
    trainer.eval()
    x_a_recon, x_ab2 = [], []
    for i in range(x_a.size(0)):
        c_a, s_a_fake = trainer.gen_a.encode(x_a[i].unsqueeze(0))
        x_ab2.append(trainer.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
    x_ab2 = torch.cat(x_ab2)
    trainer.train()
    return x_ab2


def get_Z_pushed_loader_stats(trainer, loader, device, batch_size=8, verbose=False, n_epochs=1):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=True).to(device)
    freeze(model)

    pred_arr = []

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                s_b2 = Variable(torch.randn(len(X), trainer.style_dim, 1, 1).to(trainer.device))
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = make_prediction(X[start:end].type(torch.FloatTensor).to(device),
                                            s_b2[start:end].type(torch.FloatTensor).to(device),
                                            trainer).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end - start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma
