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
subj = 8
subj_list = [subj]
num_session = 30
batch_size = 32
test_batch_size = 3000
num_samples_per_epoch = 750 * num_session
num_iterations_per_epoch = num_samples_per_epoch // batch_size
num_epochs = 300
hidden_dim = 256
n_blocks = 2

output_dir = f"./subj0{subj}_voxel_autoencoder_aligning_3e-4_L_ep300_h256_b2"
os.makedirs(output_dir, exist_ok=True)

def my_split_by_node(urls): return urls

train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0.." + f"{num_session - 1}" + "}.tar"
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


model = VoxelAutoEncoder(
    num_voxels=num_voxels,
    token_dim=clip_emb_dim,
    num_tokens=clip_seq_dim,
    hidden_dim=hidden_dim,
    n_blocks=n_blocks,
    drop=.15
)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
total_steps = int(np.floor(num_epochs * num_iterations_per_epoch))
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=total_steps,
    final_div_factor=1000,
    last_epoch=-1, pct_start=2 / num_epochs
)
soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs)

model.train()

# # test
# fmri = torch.randn(32, num_voxels).to(device)
# fmri_rep = voxel_encoder(fmri)
# pred_fmri = voxel_decoder(fmri_rep)
# print(fmri_rep.shape)
# print(pred_fmri.shape)


for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    losses, mse_losses, contrastive_losses, accuracy, lrs = [], [], [], [], []

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

        voxel_rep, pred_voxel = model(voxel)

        voxel_rep_norm = nn.functional.normalize(voxel_rep.flatten(1), dim=-1)
        image_rep_norm = nn.functional.normalize(image_rep.flatten(1), dim=-1)

        epoch_temp = soft_loss_temps[epoch]
        contrastive_loss = soft_clip_loss(voxel_rep_norm, image_rep_norm, temp=epoch_temp)
        contrastive_loss_value = contrastive_loss.item()

        mse_loss = nn.functional.mse_loss(pred_voxel, voxel)
        mse_loss_value = mse_loss.item()

        loss = mse_loss + contrastive_loss
        if loss.isnan().any():
            raise ValueError('NaN loss')
        loss_value = loss.item()

        loss.backward()
        optimizer.step()

        losses.append(loss_value)
        mse_losses.append(mse_loss_value)
        contrastive_losses.append(contrastive_loss_value)

        lrs.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()

    # test_image = None
    # test_voxel = None
    # model.eval()
    # with torch.no_grad():
    #     for behav, _, _, _ in test_dl:
    #         if test_image is None:
    #             voxel = voxels[behav[:, 0, 5].cpu().long()]
    #             image_idx = behav[:, 0, 0].cpu().long()
    #             unique_image, sort_indices = torch.unique(image_idx, return_inverse=True)
    #             for im in unique_image:
    #                 locs = torch.where(im == image_idx)[0]
    #                 if len(locs) == 1:
    #                     locs = locs.repeat(3)
    #                 elif len(locs) == 2:
    #                     locs = locs.repeat(2)[:3]
    #                 assert len(locs) == 3
    #                 if test_image is None:
    #                     test_image = torch.Tensor(images[im][None])
    #                     test_voxel = voxel[locs][None]
    #                 else:
    #                     test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
    #                     test_voxel = torch.vstack((test_voxel, voxel[locs][None]))
    #
    #     test_voxel_mean = torch.mean(test_voxel, dim=1)
    #
    #     random_samps = np.random.choice(np.arange(len(test_voxel_mean)), size=300, replace=False)
    #     test_image = test_image[random_samps].to(device)
    #     test_voxel_mean = test_voxel_mean[random_samps].to(device)
    #
    #     voxel_rep, pred_voxel = model(test_voxel_mean)
    #     if embedder_name == "ViT-bigG/14":
    #         image_rep = clip_img_embedder(test_image)
    #     elif embedder_name == "ViT-L/14":
    #         image_rep = clip_img_embedder.embed_image(test_image).float()
    #
    #     voxel_rep = nn.functional.normalize(voxel_rep.reshape(len(voxel_rep), -1), dim=-1)
    #     image_rep = nn.functional.normalize(image_rep.reshape(len(image_rep), -1), dim=-1)
    #
    #     labels = torch.arange(len(voxel_rep)).to(device)
    #     bwd_sim = batchwise_cosine_similarity(image_rep, voxel_rep)
    #     fwd_sim = batchwise_cosine_similarity(voxel_rep, image_rep)
    #
    #     assert len(bwd_sim) == 300
    #     percent_correct_fwds = topk(fwd_sim, labels, k=1).item()
    #     percent_correct_bwds = topk(bwd_sim, labels, k=1).item()
    #
    #     test_mse = nn.functional.mse_loss(pred_voxel, test_voxel_mean)
    #
    #     print("epoch %d, loss: %f, contr_loss: %f, mes_loss: %f, avg lr: %f, fwd: %f, bwd: %f, test_mse: %f" % (epoch+1, np.mean(losses), np.mean(contrastive_losses), np.mean(mse_losses), np.mean(lrs), percent_correct_fwds, percent_correct_bwds, test_mse))

    print("epoch %d, loss: %f, contr_loss: %f, mes_loss: %f, avg lr: %f" % (epoch+1, np.mean(losses), np.mean(contrastive_losses), np.mean(mse_losses), np.mean(lrs)))

    torch.save(model.state_dict(), f"{output_dir}/ckpt_{epoch+1}.pt")









