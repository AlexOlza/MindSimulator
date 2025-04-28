import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from torch.utils.data import DataLoader
import utils

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

torch.backends.cuda.matmul.allow_tf32 = True
data_type = torch.float16
device = "cuda:0"
utils.seed_everything(42)
data_path=os.getcwd()
cache_dir=os.getcwd()
subj=1
num_sessions=40
use_prior=True
test_batch_size=360
blurry_recon=False
clip_scale=1.0
prior_scale=30
new_test=True
n_blocks=2
hidden_dim=256
subj_list = [1]

def my_split_by_node(urls): return urls
num_voxels_list = []
num_voxels = {}
voxels = {}
for s in subj_list:
    f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas

if not new_test:  # using old test set from before full dataset released (used in original MindEye paper)
    if subj == 3:
        num_test = 2113
    elif subj == 4:
        num_test = 1985
    elif subj == 6:
        num_test = 2113
    elif subj == 8:
        num_test = 1985
    else:
        num_test = 2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
elif new_test:  # using larger test set from after full dataset released
    if subj == 3:
        num_test = 2371
    elif subj == 4:
        num_test = 2188
    elif subj == 6:
        num_test = 2371
    elif subj == 8:
        num_test = 2188
    else:
        num_test = 3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
    .shuffle(750, initial=1500, rng=random.Random(42)) \
    .decode("torch") \
    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
            olds_behav="olds_behav.npy") \
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f'{cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)
    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
        data_keys=["input"],
    )

class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x
model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out

model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)

from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim,
                          blurry_recon=blurry_recon, clip_scale=clip_scale)

if use_prior:
    from models import *
    out_dim = clip_emb_dim
    depth = 3
    dim_head = 52
    heads = clip_emb_dim // 52
    timesteps = 100
    prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb"
    )
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )


ckpt = torch.load("train_logs/train_subj1_001/ckpt_31.pth")
model.load_state_dict(ckpt["model_state_dict"])

model.to(device)
torch.cuda.empty_cache()
test_image, test_voxel = None, None

test_losses = []
best_test_loss = 1e9
test_fwd_percent_correct = 0.
test_bwd_percent_correct = 0.
test_recon_cossim = 0.
test_recon_mse = 0.
test_loss_clip_total = 0.
test_loss_prior_total = 0.
test_blurry_pixcorr = 0.

model.eval()
with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
    for test_i, (behav, _, _, _) in enumerate(test_dl):
        # assert len(behav) == num_test
        if test_image is None:
            voxel = voxels[f'subj0{subj}'][behav[:, 0, 5].cpu().long()].unsqueeze(1)
            image = behav[:, 0, 0].cpu().long()
            unique_image, sort_indices = torch.unique(image, return_inverse=True)
            for im in unique_image:
                locs = torch.where(im == image)[0]
                if len(locs) == 1:
                    locs = locs.repeat(3)
                elif len(locs) == 2:
                    locs = locs.repeat(2)[:3]
                assert len(locs) == 3
                if test_image is None:
                    test_image = torch.Tensor(images[im][None])
                    test_voxel = voxel[locs][None]
                else:
                    test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
                    test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

        loss = 0.0

        test_indices = torch.arange(len(test_voxel))[:300]  # test_indices = torch.arange(len(test_voxel))[:300]
        voxel = test_voxel[test_indices].to(device)
        image = test_image[test_indices].to(device)
        # assert len(image) == 300
        print("the length of test batch is %d" % len(image))

        clip_target = clip_img_embedder(image.float())

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:, rep], 0)
            backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
            if rep == 0:
                clip_voxels = clip_voxels0
                backbone = backbone0
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
        clip_voxels /= 3
        backbone /= 3

        if clip_scale > 0:
            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

        # for some evals, only doing a subset of the samples per batch because of computational cost
        random_samps = np.random.choice(np.arange(len(image)), size=len(image) // 5, replace=False)

        if use_prior:
            loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
            test_loss_prior_total += loss_prior.item()
            loss_prior *= prior_scale
            loss += loss_prior

        if clip_scale > 0:
            loss_clip = utils.soft_clip_loss(
                clip_voxels_norm,
                clip_target_norm,
                temp=.006)

            test_loss_clip_total += loss_clip.item()
            loss_clip = loss_clip * clip_scale
            loss += loss_clip

        if blurry_recon:
            image_enc_pred, _ = blurry_image_enc_
            blurry_recon_images = (
                        autoenc.decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
            test_blurry_pixcorr += pixcorr.item()

        if clip_scale > 0:
            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
            test_fwd_percent_correct += utils.topk(
                utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
            test_bwd_percent_correct += utils.topk(
                utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

        utils.check_loss(loss)
        test_losses.append(loss.item())

    # assert (test_i + 1) == 1
    logs = {"test/loss": np.mean(test_losses[-(test_i + 1):]),
            "test/num_steps": len(test_losses),
            "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
            "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
            "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
            "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
            "test/recon_cossim": test_recon_cossim / (test_i + 1),
            "test/recon_mse": test_recon_mse / (test_i + 1),
            "test/loss_prior": test_loss_prior_total / (test_i + 1),
            }

    print(logs)




