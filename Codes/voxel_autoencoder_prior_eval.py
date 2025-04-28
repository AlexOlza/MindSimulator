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
from mindeye2_src.utils import seed_everything

seed_everything(42)
embedder_name = "ViT-L/14"
# embedder_name = "ViT-bigG/14"
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda:7')
data_path = os.getcwd() + "/mindeye2_src"
cache_dir = os.getcwd() + "/mindeye2_src"
print(data_path)
new_test = True
subj = 7
subj_list = [subj]
timesteps = 100
sampling_steps = 100
drop_prob = 0.2
repeat = 5

voxel_autoencoder_path = f"./subj0{subj}_voxel_autoencoder_aligning_3e-4_L_ep300_h256_b2/ckpt_300.pt"
# voxel_autoencoder_path = "./voxel_autoencoder_aligning_2stages_3e-4_L_ep300_h256_b2/last.pt"
output_dir = f"./subj0{subj}_voxel_diffusion_prior_3e-4_L_ep150_6l_ls300_ts100_drop0.2"
voxel_diffusion_prior_path = output_dir + "/ckpt_149.pt"

def my_split_by_node(urls): return urls

# train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0..39" + "}.tar"
# train_data = wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node) \
#     .shuffle(750, initial=1500, rng=random.Random(42)) \
#     .decode("torch") \
#     .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
#             olds_behav="olds_behav.npy") \
#     .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
# train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

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
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
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
diffusion_prior.load_state_dict(torch.load(voxel_diffusion_prior_path, map_location="cpu"))
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

voxel_autoencoder.eval()
diffusion_prior.eval()
mse = nn.MSELoss()
torch.cuda.empty_cache()
test_image = None
test_voxel = None
mse_list = []
predicted_fmri = torch.zeros((1000, num_voxels)).to(device)

with torch.no_grad():
    for behav, _, _, _ in test_dl:
        if test_image is None:
            voxel = voxels[behav[:, 0, 5].cpu().long()]
            image_idx = behav[:, 0, 0].cpu().long()
            unique_image, sort_indices = torch.unique(image_idx, return_inverse=True)
            for im in unique_image:
                locs = torch.where(im==image_idx)[0]
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
    test_voxel_mean = torch.mean(test_voxel, dim=1)

    for i in tqdm(range(1000)):
        image_i = test_image[i].unsqueeze(0).to(device)
        voxel_i = test_voxel_mean[i].unsqueeze(0).to(device)

        if embedder_name == "ViT-bigG/14":
            image_rep_i = clip_img_embedder(image_i.float())
        elif embedder_name == "ViT-L/14":
            image_rep_i = clip_img_embedder.embed_image(image_i).float()

        pred_i_repeat = torch.zeros((repeat, 1, num_voxels)).to(device)
        for repe in range(repeat):
            pred_rep_i_repe = diffusion_prior.p_sample_loop([1, 257, 768], text_cond=dict(text_embed=image_rep_i),
                                                    cond_scale=1., timesteps=sampling_steps)
            pred_i_repe = voxel_autoencoder.voxel_decoder(pred_rep_i_repe)
            pred_i_repeat[repe] = pred_i_repe
        pred_i = torch.mean(pred_i_repeat, dim=0)
        predicted_fmri[i] = pred_i
        mse_list.append(mse(pred_i, voxel_i).item())

print("avg mse is %f" % np.mean(mse_list))
print(predicted_fmri.shape)
torch.save(predicted_fmri, os.path.join(output_dir, f"predicted_fmri_ep150_step{sampling_steps}_repeat{repeat}.pt"))


