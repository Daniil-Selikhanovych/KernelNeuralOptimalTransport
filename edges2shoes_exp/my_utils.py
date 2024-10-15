
import os
from shutil import copyfile
import torch
from torch.autograd import Variable
import torchvision.utils as vutils

import numpy as np

from inception import InceptionV3

import shutil
import glob
import json
import gc


def save_results(expr_dir, results_dict):
    # save to results.json (for cluster exp)
    fname = os.path.join(expr_dir, 'results.json')
    with open(fname, 'w') as f:
        json.dump(results_dict, f, indent=4)


def copy_scripts_to_folder(expr_dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in glob.glob("%s/*.py" % dir_path):
        shutil.copy(f, expr_dir)


def print_log(out_f, message):
    out_f.write(message+"\n")
    out_f.flush()
    print(message)


def format_log(epoch, i, errors, t, prefix=True):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def visualize_cycle(opt, real_A, visuals, eidx, uidx, train):
    size = real_A.size()

    images = [img.cpu().unsqueeze(1) for img in visuals.values()]
    vis_image = torch.cat(images, dim=1).view(size[0]*len(images), size[1], size[2], size[3])
    if train:
        save_path = opt.train_vis_cycle
    else:
        save_path = opt.vis_cycle
    save_path = os.path.join(save_path, 'cycle_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_image.cpu(), save_path, normalize=True, range=(-1, 1), nrow=len(images))
    copyfile(save_path, os.path.join(opt.vis_latest, 'cycle.png'))


def visualize_multi(opt, real_A, model, eidx, uidx):
    size = real_A.size()
    # all samples in real_A share the same prior_z_B
    multi_prior_z_B = Variable(real_A.data.new(opt.num_multi,
                                               opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0], 1, 1, 1),
                               volatile=True)
    multi_fake_B = model.generate_multi(real_A.detach(), multi_prior_z_B)
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])
    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1), size[1], size[2], size[3])
    save_path = os.path.join(opt.vis_multi, 'multi_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_multi_image.cpu(), save_path, normalize=True,
                      range=(-1, 1), nrow=opt.num_multi+1)
    copyfile(save_path, os.path.join(opt.vis_latest, 'multi.png'))


def visualize_inference(opt, real_A, real_B, model, eidx, uidx):
    size = real_A.size()

    real_B = real_B[:opt.num_multi]
    # all samples in real_A share the same post_z_B
    multi_fake_B = model.inference_multi(real_A.detach(), real_B.detach())
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])

    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1), size[1], size[2], size[3])

    vis_multi_image = torch.cat([torch.ones(1, size[1], size[2], size[3]).cpu(), real_B.data.cpu(),
                                 vis_multi_image.cpu()], dim=0)

    save_path = os.path.join(opt.vis_inf, 'inf_%02d_%04d.png' % (eidx, uidx))
    vutils.save_image(vis_multi_image.cpu(), save_path, normalize=True,
                      range=(-1, 1), nrow=opt.num_multi+1)
    copyfile(save_path, os.path.join(opt.vis_latest, 'inf.png'))


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def get_Z_pushed_loader_stats(gan_model, loader, opt, device, batch_size=8):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=True).to(device)
    freeze(model)
    pred_arr = []

    gan_model.eval()

    with torch.no_grad():
        for step, (X, _) in enumerate(loader):
            prior_z_B_current = Variable(torch.randn(X.size(0), opt.nlatent, 1, 1))
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                batch = gan_model.netG_A_B.forward(X[start:end].type(torch.FloatTensor).to(device),
                                                   prior_z_B_current[start:end].type(torch.FloatTensor).to(device))

                batch = batch.add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end - start, -1))

                model.eval()

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma