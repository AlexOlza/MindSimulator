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
from accelerate import Accelerator, DeepSpeedPlugin

from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import evaluate
import pandas as pd

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from models import GNet8_Encoder

torch.backends.cuda.matmul.allow_tf32 = True
import utils

utils.seed_everything(42)

data_path = os.getcwd()
cache_dir = os.getcwd()

subj_list = [1]

device = "cuda:7"

voxel_level = False
pix_corr = True
ssim = True
alexnet = True
inception3 = False
clip = False
eff = False
swav = False


mse_v = 0
pearsonr_v = 0
r2_v = 0
pix_corr_v = 0
ssim_v = 0
alexnet2_v = 0
alexnet5_v = 0
inception3_v = 0
clip_v = 0
eff_v = 0
swav_v = 0


with torch.no_grad():
    for subj in subj_list:

        if subj == 1:
            num_voxels = 15724
        elif subj == 2:
            num_voxels = 14278
        elif subj == 5:
            num_voxels = 13039
        elif subj == 7:
            num_voxels = 12682

        all_images = torch.load("./all_images.pt", map_location=device)
        all_recons = torch.load(f"../random_input/memory_input/all_recons.pth", map_location=device)
        all_fmri = torch.load(f"../subj0{subj}_voxel_diffusion_prior_3e-4_L_ep150_6l_ls300_ts100_drop0.2/predicted_fmri_ep150_step100_repeat5.pt", map_location=device)


        if voxel_level:
            def my_split_by_node(urls): return urls

            new_test = True
            f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
            betas = f['betas'][:]
            betas = torch.Tensor(betas).to("cpu").to(torch.float16)
            num_voxels = betas[0].shape[-1]
            voxels = betas

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


            test_voxel = None
            mse_f = torch.nn.MSELoss()
            from scipy.stats import pearsonr

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    for behav, _, _, _ in test_dl:
                        if test_voxel is None:
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
                                if test_voxel is None:
                                    test_voxel = voxel[locs][None]
                                else:
                                    test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                    test_voxel_mean = torch.mean(test_voxel, dim=1)
                    test_voxel_mean = test_voxel_mean.to(device)

                    mse = mse_f(test_voxel_mean, all_fmri).item()
                    print(mse)

                    mse_v += mse

                    pear_list = []
                    for i in range(test_voxel_mean.shape[0]):
                        test_voxel_i = test_voxel_mean[i]
                        pred_voxel_i = all_fmri[i]
                        pear, _ = pearsonr(test_voxel_i.cpu().numpy(), pred_voxel_i.cpu().numpy())
                        pear_list.append(pear)
                    pearsonr = np.mean(pear_list)
                    print(pearsonr)

                    pearsonr_v += pearsonr

                    y_true_mean = torch.mean(test_voxel_mean, dim=0)
                    ss_res = torch.sum((test_voxel_mean - all_fmri) ** 2, dim=0)
                    ss_tot = torch.sum((test_voxel_mean - y_true_mean) ** 2, dim=0)
                    r2 = 1 - (ss_res / ss_tot)

                    torch.save(r2, f"../r_2/subj0{subj}.pt")

                    r2_m = torch.mean(r2).item()
                    print(r2_m)

                    r2_v += r2_m



        if pix_corr:
            preprocess = transforms.Compose([
                transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
            ])

            # Flatten images while keeping the batch dimension
            all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
            all_recons_flattened = preprocess(all_recons).view(len(all_recons), -1).cpu()
            print(all_images_flattened.shape)
            print(all_recons_flattened.shape)
            corrsum = 0
            for i in tqdm(range(len(all_images))):
                corrsum += np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
            corrmean = corrsum / len(all_images)
            pixcorr = corrmean
            print(pixcorr)

            pix_corr_v += pixcorr


        if ssim:
            from skimage.color import rgb2gray
            from skimage.metrics import structural_similarity as ssim

            preprocess = transforms.Compose([
                transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
            ])

            # convert image to grayscale with rgb2grey
            img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
            recon_gray = rgb2gray(preprocess(all_recons).permute((0, 2, 3, 1)).cpu())
            print("converted, now calculating ssim...")

            ssim_score = []
            for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images)):
                ssim_score.append(
                    ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                         data_range=1.0))

            ssim = np.mean(ssim_score)
            print(ssim)

            ssim_v += ssim


        from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
        @torch.no_grad()
        def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
            preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
            reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
            if feature_layer is None:
                preds = preds.float().flatten(1).cpu().numpy()
                reals = reals.float().flatten(1).cpu().numpy()
            else:
                preds = preds[feature_layer].float().flatten(1).cpu().numpy()
                reals = reals[feature_layer].float().flatten(1).cpu().numpy()
            r = np.corrcoef(reals, preds)
            r = r[:len(all_images), len(all_images):]
            congruents = np.diag(r)
            success = r < congruents
            success_cnt = np.sum(success, 0)
            if return_avg:
                perf = np.mean(success_cnt) / (len(all_images)-1)
                return perf
            else:
                return success_cnt, len(all_images)-1


        if alexnet:
            from torchvision.models import alexnet, AlexNet_Weights

            alex_weights = AlexNet_Weights.IMAGENET1K_V1

            alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4', 'features.11']).to(
                device)
            alex_model.eval().requires_grad_(False)
            # see alex_weights.transforms()
            preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            layer = 'early, AlexNet(2)'
            print(f"\n---{layer}---")
            all_per_correct = two_way_identification(all_recons.to(device).float(), all_images, alex_model, preprocess, 'features.4')
            alexnet2 = np.mean(all_per_correct)
            print(f"2-way Percent Correct: {alexnet2:.4f}")

            alexnet2_v += alexnet2

            layer = 'mid, AlexNet(5)'
            print(f"\n---{layer}---")
            all_per_correct = two_way_identification(all_recons.to(device).float(), all_images, alex_model, preprocess, 'features.11')
            alexnet5 = np.mean(all_per_correct)
            print(f"2-way Percent Correct: {alexnet5:.4f}")

            alexnet5_v += alexnet5


        if inception3:
            from torchvision.models import inception_v3, Inception_V3_Weights

            weights = Inception_V3_Weights.DEFAULT
            inception_model = create_feature_extractor(inception_v3(weights=weights),
                                                       return_nodes=['avgpool']).to(device)
            inception_model.eval().requires_grad_(False)
            # see weights.transforms()
            preprocess = transforms.Compose([
                transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            all_per_correct = two_way_identification(all_recons, all_images,
                                                     inception_model, preprocess, 'avgpool')
            inception = np.mean(all_per_correct)
            print(f"2-way Percent Correct: {inception:.4f}")

            inception3_v += inception


        if clip:
            import clip
            clip_model, preprocess = clip.load("ViT-L/14", device=device)
            preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711]),
            ])

            all_per_correct = two_way_identification(all_recons, all_images,
                                                     clip_model.encode_image, preprocess, None)  # final layer
            clip_ = np.mean(all_per_correct)
            print(f"2-way Percent Correct: {clip_:.4f}")

            clip_v += clip_


        if eff:
            import scipy as sp
            from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
            weights = EfficientNet_B1_Weights.DEFAULT
            eff_model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=['avgpool']).to(device)
            eff_model.eval().requires_grad_(False)
            # see weights.transforms()
            preprocess = transforms.Compose([
                transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            gt = eff_model(preprocess(all_images))['avgpool']
            gt = gt.reshape(len(gt), -1).cpu().numpy()
            fake = eff_model(preprocess(all_recons))['avgpool']
            fake = fake.reshape(len(fake), -1).cpu().numpy()
            effnet = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean()
            print("Distance:", effnet)

            eff_v += effnet


        if swav:
            import scipy as sp
            swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            swav_model = create_feature_extractor(swav_model, return_nodes=['avgpool']).to(device)
            swav_model.eval().requires_grad_(False)
            preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            gt = swav_model(preprocess(all_images))['avgpool']
            gt = gt.reshape(len(gt), -1).cpu().numpy()
            fake = swav_model(preprocess(all_recons))['avgpool']
            fake = fake.reshape(len(fake), -1).cpu().numpy()
            swav = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean()
            print("Distance:", swav)

            swav_v += swav


print("")
print("")
print("")
print("")
print("")
print("")

mse_v /= len(subj_list)
pearsonr_v /= len(subj_list)
r2_v /= len(subj_list)
pix_corr_v /= len(subj_list)
ssim_v /= len(subj_list)
alexnet2_v /= len(subj_list)
alexnet5_v /= len(subj_list)
inception3_v /= len(subj_list)
clip_v /= len(subj_list)
eff_v /= len(subj_list)
swav_v /= len(subj_list)


print(mse_v)
print(pearsonr_v)
print(r2_v)
print(pix_corr_v)
print(ssim_v)
print(alexnet2_v)
print(alexnet5_v)
print(inception3_v)
print(clip_v)
print(eff_v)
print(swav_v)
