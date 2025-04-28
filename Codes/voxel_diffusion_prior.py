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
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from torch.utils.data import DataLoader
sys.path.append("mindeye2_src/")
from mindeye2_src.generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from mindeye2_src.models import Clipper
from mindeye2_src.models import BrainDiffusionPrior, PriorNetwork, VoxelAutoEncoder

embedder_name = "ViT-L/14"
# embedder_name = "ViT-bigG/14"
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda:7')
data_path = os.getcwd() + "/mindeye2_src"
cache_dir = os.getcwd() + "/mindeye2_src"
print(data_path)
new_test = True
subj = 8
subj_list = [subj]
num_sessions = 30
batch_size = 32
test_batch_size = 32
num_samples_per_epoch = 750 * num_sessions
num_iterations_per_epoch = num_samples_per_epoch // batch_size
num_epochs = 150
loss_scale = 300.
timesteps = 100
drop_prob = 0.2
voxel_autoencoder_path = f"./subj0{subj}_voxel_autoencoder_aligning_3e-4_L_ep300_h256_b2/ckpt_300.pt"
# voxel_autoencoder_path = "./voxel_autoencoder_aligning_2stages_3e-4_L_ep300_h256_b2/last.pt"

output_dir = f"./subj0{subj}_voxel_diffusion_prior_3e-4_L_ep150_6l_ls300_ts100_drop0.2"
os.makedirs(output_dir, exist_ok=True)

def my_split_by_node(urls): return urls

train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
train_data = wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node) \
    .shuffle(750, initial=1500, rng=random.Random(42)) \
    .decode("torch") \
    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
            olds_behav="olds_behav.npy") \
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])

train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu").float()
num_voxels = betas[0].shape[-1]
voxels = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")
print(f"Loaded train dl and betas for subj0{subj}!")


if not new_test:
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
print(f"Loaded test dl for subj{subj}!")


f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']
print("Loaded all 73k possible NSD images to cpu!", images.shape)

if embedder_name == "ViT-bigG/14":
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder = clip_img_embedder.to(device)
    clip_seq_dim = 256
    clip_emb_dim = 1664

if embedder_name == "ViT-L/14":
    clip_img_embedder = Clipper(
        "ViT-L/14",
        device=device,
        hidden_state=True,
        norm_embs=True
    )
    clip_img_embedder = clip_img_embedder.to(device)
    clip_seq_dim = 257
    clip_emb_dim = 768

out_dim = clip_emb_dim
depth = 6
if embedder_name == "ViT-bigG/14":
    dim_head = 52
elif embedder_name == "ViT-L/14":
    dim_head = 48
heads = clip_emb_dim // 52

prior = PriorNetwork(
    dim=out_dim,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    causal=False,
    num_tokens=clip_seq_dim,
    learned_query_mode="pos_emb"
)
diffusion_prior = BrainDiffusionPrior(
    net=prior,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=drop_prob,
    image_embed_scale=None,
)
diffusion_prior = diffusion_prior.to(device)

voxel_autoencoder = VoxelAutoEncoder(
    num_voxels=num_voxels,
    token_dim=clip_emb_dim,
    num_tokens=clip_seq_dim,
    hidden_dim=256,
    n_blocks=2,
    drop=.15
)
voxel_autoencoder.load_state_dict(torch.load(voxel_autoencoder_path, map_location="cpu"))
voxel_autoencoder = voxel_autoencoder.to(device)
voxel_autoencoder.requires_grad_(False)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 1e-2},
    {'params': [p for n, p in diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=3e-4)
total_steps = int(np.floor(num_epochs * num_iterations_per_epoch))
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=total_steps,
    final_div_factor=1000,
    last_epoch=-1, pct_start=2 / num_epochs
)

torch.cuda.empty_cache()
diffusion_prior.train()

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs))
    train_losses, mse, lrs = [], [], []

    image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224).float()
    voxel_iters = torch.zeros(num_iterations_per_epoch, batch_size, num_voxels).float()

    iter = -1
    for behav0, _, _, _ in train_dl:
        image_idx = behav0[:, 0, 0].cpu().long().numpy()
        image0, image_sorted_idx = np.unique(image_idx, return_index=True)
        if len(image0) != len(image_idx):
            continue
        iter += 1
        image0 = torch.tensor(images[image0])
        image_iters[iter] = image0

        voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
        voxel_sorted_idx = voxel_idx[image_sorted_idx]
        voxel0 = voxels[voxel_sorted_idx]
        voxel0 = torch.Tensor(voxel0)
        voxel_iters[iter] = voxel0

        if iter >= num_iterations_per_epoch - 1:
            break

    for step in tqdm(range(num_iterations_per_epoch)):
        optimizer.zero_grad()

        voxel = voxel_iters[step].detach().to(device)
        image = image_iters[step].detach().to(device)

        with torch.no_grad():
            if embedder_name == "ViT-bigG/14":
                image_rep = clip_img_embedder(image)
            elif embedder_name == "ViT-L/14":
                image_rep = clip_img_embedder.embed_image(image).float()
            assert not torch.any(torch.isnan(image_rep))

        with torch.no_grad():
            voxel_rep = voxel_autoencoder.voxel_encoder(voxel)
            assert not torch.any(torch.isnan(voxel_rep))

        loss, _ = diffusion_prior(text_embed=image_rep, image_embed=voxel_rep)
        if loss.isnan().any():
            raise ValueError('NaN loss')

        mse_value = loss.item()
        loss *= loss_scale
        loss_value = loss.item()
        mse.append(mse_value)
        train_losses.append(loss_value)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion_prior.parameters(), max_norm=2.0)
        optimizer.step()

        lr_value = optimizer.param_groups[0]['lr']
        lrs.append(lr_value)

        lr_scheduler.step()

    print("epoch %d, train loss: %f, lrs: %f mse: %f" % (epoch, np.mean(train_losses), np.mean(lrs), np.mean(mse)))
    torch.save(diffusion_prior.state_dict(), f"{output_dir}/ckpt_{epoch}.pt")



