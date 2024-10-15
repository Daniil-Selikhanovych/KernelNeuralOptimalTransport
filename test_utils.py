import os
import h5py
import torch.nn.functional as F
import numpy as np

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, \
    RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from core.inception import InceptionV3
from tqdm import tqdm


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


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


@torch.no_grad()
def get_Z_pushed_loader_stats(nets, domains, args, device, loader_src, n_epochs=1):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=True).to(device)
    freeze(model)
    pred_arr = []

    print('Calculating evaluation metrics...')

    eval_trg_domain = domains['target']
    eval_src_domain = domains['source']

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    for epoch in range(n_epochs):
        for trg_idx, trg_domain in enumerate(domains):
            src_domains = [x for x in domains if x != trg_domain]
            for src_idx, src_domain in enumerate(src_domains):
                if src_domain == eval_src_domain and trg_domain == eval_trg_domain:
                    task = '%s2%s' % (src_domain, trg_domain)
                    print(f"Compute FID for {task}")
                    for i, data in enumerate(tqdm(loader_src, total=len(loader_src))):
                        x_src = data[0]
                        N = x_src.size(0)
                        x_src = x_src.to(device)
                        y_trg = torch.tensor([trg_idx] * N).to(device)
                        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                        z_trg = torch.randn(N, args.latent_dim).to(device)
                        s_trg = nets.mapping_network(z_trg, y_trg)
                        x_fake = nets.generator(x=x_src, s=s_trg, masks=masks)
                        x_fake = x_fake.add(1).mul(.5)
                        pred_arr.append(model(x_fake)[0].cpu().data.numpy().reshape(N, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    torch.cuda.empty_cache()
    return mu, sigma
