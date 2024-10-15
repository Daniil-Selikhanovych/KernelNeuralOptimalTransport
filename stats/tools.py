import gc
import numpy as np
import torch

from inception import InceptionV3

from tqdm import tqdm_notebook as tqdm


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


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


def get_loader_stats(loader, batch_size=8, verbose=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=True).cuda()
    freeze(model)

    size = len(loader.dataset)
    pred_arr = []

    with torch.no_grad():
        for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end - start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect();
    torch.cuda.empty_cache()
    return mu, sigma