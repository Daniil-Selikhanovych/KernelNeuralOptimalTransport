import os
from os.path import join as ospj
import time
import datetime
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
import core.utils as utils
from core.my_data_loader import InputFetcher

from munch import Munch
from core.my_metrics import get_Z_pushed_loader_stats
from core.fid_score import calculate_frechet_distance

from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


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


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

        self.BEST_FID = np.inf
        self.best_iter = 0
        filename = 'stats/{}_{}_test.json'.format(args.target_dataset, args.img_size)
        with open(filename, 'r') as fp:
            data_stats = json.load(fp)
            self.mu_data, self.sigma_data = data_stats['mu'], data_stats['sigma']
        del data_stats

        self.path_to_writer = os.path.join(args.OUTPUT_PATH, "tensorboard")
        self.writer = SummaryWriter(self.path_to_writer)
        print(f"create writer in {self.path_to_writer}")

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        transform = Compose([Resize((args.img_size, args.img_size)),
                             ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset_a = ImageFolder(args.train_a, transform)
        train_dataset_b = ImageFolder(args.train_b, transform)
        test_dataset_a = ImageFolder(args.test_a, transform)
        test_dataset_b = ImageFolder(args.test_b, transform)

        train_loader_a = DataLoader(dataset=train_dataset_a,
                                    batch_size=20,
                                    num_workers=4,
                                    pin_memory=True,
                                    shuffle=False)

        train_loader_b = DataLoader(dataset=train_dataset_b,
                                    batch_size=20,
                                    num_workers=4,
                                    pin_memory=True,
                                    shuffle=False)

        test_loader_a = DataLoader(dataset=test_dataset_a,
                                   batch_size=20,
                                   num_workers=4,
                                   pin_memory=True,
                                   shuffle=False)

        test_loader_b = DataLoader(dataset=test_dataset_b,
                                   batch_size=20,
                                   num_workers=4,
                                   pin_memory=True,
                                   shuffle=False)

        X_sampler = LoaderSampler(train_loader_a)
        X_test_sampler = LoaderSampler(test_loader_a)
        Y_sampler = LoaderSampler(train_loader_b)
        Y_test_sampler = LoaderSampler(test_loader_b)


        torch.manual_seed(0xBADBEEF)
        np.random.seed(0xBADBEEF)
        self.X_fixed = X_sampler.sample(10)
        self.Y_fixed = Y_sampler.sample(10)

        self.X_test_fixed = X_test_sampler.sample(10)
        self.Y_test_fixed = Y_test_sampler.sample(10)

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                print('Computing FID')
                mu, sigma = get_Z_pushed_loader_stats(nets_ema, args.domains, args, args.device,
                                                      batch_size=8, n_epochs=args.n_epochs)
                fid = calculate_frechet_distance(self.mu_data, self.sigma_data, mu, sigma)
                print(f"FID = {fid}, best FID = {self.BEST_FID}, best iter = {self.best_iter}")
                self.writer.add_scalar('test fid',
                                       fid,
                                       i + 1)
                del mu, sigma

                num_examples = 4
                Y_fakes = []
                Y_test_fakes = []

                model.eval()

                for v in range(num_examples):
                    N = x_src.size(0)
                    x_src = x_src.to(device)
                    y_trg = torch.tensor([trg_idx] * N).to(device)
                    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                    z_trg = torch.randn(N, args.latent_dim).to(device)
                    s_trg = nets.mapping_network(z_trg, y_trg)
                    x_fake = nets.generator(x_src, s_trg, masks=masks)

                    fake_Y = model.netG_A_B.forward(X_fixed_variable, prior_z_B_current).detach()
                    fake_Y_numpy = tensor2img(fake_Y)
                    Y_fakes.append(fake_Y_numpy)

                    fake_Y_test = model.netG_A_B.forward(X_test_fixed_variable, prior_z_B_current).detach()
                    fake_Y_test_numpy = tensor2img(fake_Y_test)
                    Y_test_fakes.append(fake_Y_test_numpy)

                real_X_numpy = tensor2img(self.X_fixed)
                real_X_test_numpy = tensor2img(self.X_test_fixed)
                real_Y_numpy = tensor2img(self.Y_fixed)
                real_Y_test_numpy = tensor2img(self.Y_test_fixed)

                X_tensors = [real_X_numpy, real_X_test_numpy]
                Y_tensors = [real_Y_numpy, real_Y_test_numpy]

                Y_fakes_tensors = [Y_fakes, Y_test_fakes]

                modes = ["train", "test"]

                print(f"Plotting, iter = {i}")

                for t in range(2):
                    print(f"mode = {modes[t]}")
                    fig, axes = plt.subplots(6, 10, figsize=(15, 9), dpi=150)
                    X_tensor = X_tensors[t]
                    Y_tensor = Y_tensors[t]
                    Y_fake_tensor = Y_fakes_tensors[t]

                    for j in range(10):
                        axes[0][j].imshow(X_tensor[j])
                        axes[0][j].get_xaxis().set_visible(False)
                        axes[0][j].get_yaxis().set_visible(False)

                    for j in range(10):
                        axes[5][j].imshow(Y_tensor[j])
                        axes[5][j].get_xaxis().set_visible(False)
                        axes[5][j].get_yaxis().set_visible(False)

                    for i in range(4):
                        for j in range(10):
                            cur_img = Y_fake_tensor[i][j]

                            axes[i + 1][j].imshow(cur_img)
                            axes[i + 1][j].get_xaxis().set_visible(False)
                            axes[i + 1][j].get_yaxis().set_visible(False)

                    plt.show(fig)
                    self.writer.add_figure(f'Step {i + 1}, {modes[t]}',
                                           fig,
                                           global_step=i + 1)
                    plt.close(fig)

                if fid < self.BEST_FID:
                    self.BEST_FID = fid
                    self.best_iter = i

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x=x_real, s=s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x=x_real, s=s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x=x_real, s=s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x=x_fake, s=s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg