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
from mindeye2_src.models import VoxelAutoEncoder
from mindeye2_src.utils import cosine_anneal, soft_clip_loss, batchwise_cosine_similarity, topk, seed_everything

seed_everything(42)
embedder_name = "ViT-L/14"  # ViT-L/14 or ViT-bigG/14
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda:5')
data_path = os.getcwd() + "/mindeye2_src"
cache_dir = os.getcwd() + "/mindeye2_src"
print(data_path)
new_test = True
subj = 1
subj_list = [1]
batch_size = 32
test_batch_size = 3000
num_samples_per_epoch = 750*40
num_iterations_per_epoch = num_samples_per_epoch // batch_size
hidden_dim = 256
n_blocks = 4
ckpt_name = "last.pt"
file_dir = "./voxel_autoencoder_aligning_2stages_3e-4_L_ep300_h256_b4"

def my_split_by_node(urls): return urls

# train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0..39" + "}.tar"
# train_data = wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node) \
#     .shuffle(750, initial=1500, rng=random.Random(42)) \
#     .decode("torch") \
#     .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
#             olds_behav="olds_behav.npy") \
#     .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
#
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

model = VoxelAutoEncoder(
    num_voxels=num_voxels,
    token_dim=clip_emb_dim,
    num_tokens=clip_seq_dim,
    hidden_dim=hidden_dim,
    n_blocks=n_blocks,
    drop=.15
)

model.load_state_dict(torch.load(os.path.join(file_dir, ckpt_name), map_location="cpu"))
model = model.to(device)
model.eval()

test_image = None
test_voxel = None
pred_voxel_from_image = None
pred_voxel_from_voxel = None
with torch.no_grad():
    for behav, _, _, _ in test_dl:
        if test_image is None:
            voxel = voxels[behav[:, 0, 5].cpu().long()]
            image_idx = behav[:, 0, 0].cpu().long()
            unique_image, sort_indices = torch.unique(image_idx, return_inverse=True)
            for im in unique_image:
                locs = torch.where(im == image_idx)[0]
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

    test_voxel_mean = torch.mean(test_voxel, dim=1).to(device)
    test_image = test_image.to(device)

    for idx in tqdm(range(test_voxel_mean.shape[0])):
        voxel_i = test_voxel_mean[idx].unsqueeze(0)
        image_i = test_image[idx].unsqueeze(0)
        if embedder_name == "ViT-bigG/14":
            image_rep_i = clip_img_embedder(image_i)
        elif embedder_name == "ViT-L/14":
            image_rep_i = clip_img_embedder.embed_image(image_i).float()

        _, pred_voxel_from_voxel_i = model(voxel_i)
        if pred_voxel_from_voxel is None:
            pred_voxel_from_voxel = pred_voxel_from_voxel_i
        else:
            pred_voxel_from_voxel = torch.cat((pred_voxel_from_voxel, pred_voxel_from_voxel_i), dim=0)

        pred_voxel_from_image_i = model.voxel_decoder(image_rep_i)
        if pred_voxel_from_image is None:
            pred_voxel_from_image = pred_voxel_from_image_i
        else:
            pred_voxel_from_image = torch.cat((pred_voxel_from_image, pred_voxel_from_image_i), dim=0)


    mse_from_voxel = nn.functional.mse_loss(pred_voxel_from_voxel, test_voxel_mean)
    mse_from_image = nn.functional.mse_loss(pred_voxel_from_image, test_voxel_mean)
    print("mse_from_voxel: %f" % mse_from_voxel)
    print("mse_from_image: %f" % mse_from_image)
    torch.save(pred_voxel_from_voxel, os.path.join(file_dir, "pred_voxel_from_voxel.pt"))
    torch.save(pred_voxel_from_image, os.path.join(file_dir, "pred_voxel_from_image.pt"))










