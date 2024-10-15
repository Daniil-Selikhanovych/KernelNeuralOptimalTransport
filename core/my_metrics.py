import os
import numpy as np
import torch
from core.inception import InceptionV3
from tqdm import tqdm

from core.my_loader import get_test_loader
import gc
import torch.nn.functional as F


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


@torch.no_grad()
def get_Z_pushed_loader_stats(nets, domains, args, device, batch_size=8, n_epochs=1):
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
                    path_src = args.test_a
                    loader_src = get_test_loader(path_src,
                                                 img_size=args.img_size,
                                                 batch_size=batch_size,
                                                 shuffle=False)
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


def calculate_cost(nets, args, trg_idx, loader, device,
                   cost_type='mse', verbose=False):
    size = len(loader.dataset)

    cost = 0
    for step, (X, _) in tqdm(enumerate(loader)) if verbose else enumerate(loader):
        x_src = X.to(device)
        N = x_src.size(0)
        y_trg = torch.tensor([trg_idx] * N).to(device)
        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

        z_trg = torch.randn(N, args.latent_dim).to(device)
        s_trg = nets.mapping_network(z_trg, y_trg)
        x_fake = nets.generator(x=x_src, s=s_trg, masks=masks)
        if cost_type == 'mse':
            cost += (F.mse_loss(x_src, x_fake) * X.shape[0]).item()
        elif cost_type == 'l1':
            cost += (F.l1_loss(x_src, x_fake) * X.shape[0]).item()
        else:
            raise Exception('Unknown COST')
        del X, x_fake

    cost = cost / size
    gc.collect();
    torch.cuda.empty_cache()
    return cost