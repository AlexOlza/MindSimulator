import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

tensors = torch.load("./recon_results/train_subj1_001/train_subj1_001_all_recons.pt", map_location="cpu")
print(tensors.shape)

for i in tqdm(range(tensors.shape[0])):
    tensor = tensors[i]
    numpy_array = tensor.permute(1, 2, 0).numpy()
    image = Image.fromarray((numpy_array * 255).astype(np.uint8))
    image.save(f"recon_results/train_subj1_001/recon_images/{i}.png")



