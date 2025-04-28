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
from mindeye2_src.models import Clipper, BrainDiffusionPrior, PriorNetwork, VoxelAutoEncoder
from mindeye2_src.utils import seed_everything

seed_everything(42)
encoding_type = "ae+prior"
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda:0')
data_path = os.getcwd() + "/mindeye2_src"
cache_dir = os.getcwd() + "/mindeye2_src"
new_test = True
subj = 1
embedder_name = "ViT-L/14"
timesteps = 10
sampling_steps = 10
drop_prob = 0.0
repeat = 1
voxel_autoencoder_path = "./voxel_autoencoder_aligning_3e-4_L_ep300_h256_b2/ckpt_300.pt"
# voxel_autoencoder_path = "./voxel_autoencoder_aligning_2stages_3e-4_L_ep300_h256_b2/last.pt"
output_dir = "./voxel_diffusion_prior_3e-4_L_ep150_6l_ls300_ts10_drop0.0"
voxel_diffusion_prior_path = output_dir + "/ckpt_149.pt"


stimuli_set_path = "stimuli_sets_73k"
rois = ["places"]  # ["places", "bodies", "faces", "words"]
clip_name = "large14"
prompt = "prompt01"
topk = 2000
repeat = 1

output_dir = f"./conception_localization/subj0{subj}"
os.makedirs(output_dir, exist_ok=True)

def my_split_by_node(urls): return urls

if subj == 1:
    num_voxels = 15724

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

mse = nn.MSELoss()
torch.cuda.empty_cache()
voxel_autoencoder.eval()
diffusion_prior.eval()

for roi in rois:
    image = torch.load(f"./{stimuli_set_path}/{roi}_top{topk}_{clip_name}_{prompt}.pt", map_location="cpu")
    pred_fmri = None

    with torch.no_grad():

        for i in tqdm(range(image.shape[0])):
            image_i = image[i].unsqueeze(0).to(device)

            if embedder_name == "ViT-bigG/14":
                image_rep_i = clip_img_embedder(image_i.float())
            elif embedder_name == "ViT-L/14":
                image_rep_i = clip_img_embedder.embed_image(image_i).float()

            pred_fmri_i = None
            for repe in range(repeat):
                pred_fmri_rep_i_repe = diffusion_prior.p_sample_loop([1, 257, 768], text_cond=dict(text_embed=image_rep_i),
                                                                  cond_scale=1., timesteps=sampling_steps)
                pred_fmri_i_repe = voxel_autoencoder.voxel_decoder(pred_fmri_rep_i_repe)

                pred_fmri_i = pred_fmri_i_repe if pred_fmri_i is None else torch.cat((pred_fmri_i, pred_fmri_i_repe), dim=0)
            pred_fmri_i = pred_fmri_i.unsqueeze(0)
            pred_fmri = pred_fmri_i if pred_fmri is None else torch.cat((pred_fmri, pred_fmri_i), dim=0)

    print(pred_fmri.shape)
    torch.save(pred_fmri, f"./conception_localization/subj0{subj}/{roi}_top{topk}_repeat{repeat}_{clip_name}_{prompt}_{encoding_type}_pred_fmri.pt")

