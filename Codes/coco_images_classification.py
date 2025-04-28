import os
import h5py
import torch
from tqdm import tqdm
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToPILImage

device = "cuda:0"
data_path = os.getcwd() + "/mindeye2_src"
get_probs = False
topk = 200
clip_name = "large14"  # "base16", "large14"

output_dir = "./stimuli_sets_73k"
os.makedirs(output_dir, exist_ok=True)

f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
coco_images = f['images']
print("Loaded all 73k possible NSD images to cpu!", coco_images.shape)

if get_probs:

    class_labels = ["a photo of cow", "a photo of tree", "a photo of kitchen", "a photo of windows", "a photo of motorcycle", "a photo of road",
                    "a photo of airplane", "a photo of cloud", "a photo of baseball", "a photo of green", "a photo of cat", "a photo of bench"]

    if clip_name == "base16":
        model = CLIPModel.from_pretrained("./clip-vit-base-patch16").to(device)
        processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch16")
    elif clip_name == "large14":
        model = CLIPModel.from_pretrained("./clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("./clip-vit-large-patch14")

    all_probs = None
    for idx in tqdm(range(73000)):
        image = torch.tensor(coco_images[idx]).unsqueeze(0).to(device)
        # faces, words, bodies, places
        inputs = processor(text=class_labels, return_tensors="pt", padding=True)
        text = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=text, pixel_values=image, attention_mask=attention_mask)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        all_probs = probs if all_probs is None else torch.cat((all_probs, probs), dim=0)

    torch.save(all_probs, os.path.join(output_dir, f"./coco_images_probs_{clip_name}_prompt_extra1.pt"))

    # check top-10 image
    os.makedirs(os.path.join(output_dir, f"top10_visualization_{clip_name}_prompt_extra1"), exist_ok=True)

    # top_10_faces_indices = torch.argsort(all_probs[:, 0])[-10:]
    # top_10_words_indices = torch.argsort(all_probs[:, 1])[-10:]
    # top_10_bodies_indices = torch.argsort(all_probs[:, 2])[-10:]
    # top_10_places_indices = torch.argsort(all_probs[:, 3])[-10:]
    #
    # to_pil_image = ToPILImage()
    # for indices, roi in zip([top_10_faces_indices, top_10_words_indices, top_10_bodies_indices, top_10_places_indices],["faces", "words", "bodies", "places"]):
    #     for k in range(len(indices)):
    #         image = torch.tensor(coco_images[indices[k]])
    #         image = to_pil_image(image)
    #         image.save(os.path.join(output_dir, f"top10_visualization_{clip_name}_prompt_extra1", f"{roi}-{k}.png"))

    exit(0)

all_probs = torch.load(os.path.join(output_dir, f"./coco_images_probs_{clip_name}_prompt_extra1.pt"), map_location="cpu")

rois = ["cow", "tree", "kitchen", "windows", "motorcycle", "road", "airplane", "cloud", "baseball", "green", "cat", "bench"]

for roi_idx, roi in enumerate(rois):
    topk_k_roi_indices = torch.argsort(all_probs[:, roi_idx])[-topk:]
    stimuli_set = None
    for j in range(len(topk_k_roi_indices)):
        stimuli =  torch.tensor(coco_images[topk_k_roi_indices[j]]).unsqueeze(0)
        stimuli_set = stimuli if stimuli_set is None else torch.cat((stimuli_set, stimuli), dim=0)
    print(stimuli_set.shape)
    torch.save(stimuli_set, os.path.join(output_dir, f"{roi}_top{topk}_{clip_name}_prompt_extra1.pt"))

    os.makedirs(os.path.join(output_dir, f"top{topk}_visualization_{clip_name}_prompt_extra1"), exist_ok=True)
    to_pil_image = ToPILImage()
    for k in range(len(topk_k_roi_indices)):
        image = torch.tensor(coco_images[topk_k_roi_indices[k]])
        image = to_pil_image(image)
        image.save(os.path.join(output_dir, f"top{topk}_visualization_{clip_name}_prompt_extra1", f"{roi}-{k}.png"))


# top_k_faces_indices = torch.argsort(all_probs[:, 0])[-topk:]
# top_k_words_indices = torch.argsort(all_probs[:, 1])[-topk:]
# top_k_bodies_indices = torch.argsort(all_probs[:, 2])[-topk:]
# top_k_places_indices = torch.argsort(all_probs[:, 3])[-topk:]
#
# for indices, roi in zip([top_k_faces_indices, top_k_words_indices, top_k_bodies_indices, top_k_places_indices], ["faces", "words", "bodies", "places"]):
#     stimuli_set = None
#     for j in range(len(indices)):
#         stimuli =  torch.tensor(coco_images[indices[j]]).unsqueeze(0)
#         stimuli_set = stimuli if stimuli_set is None else torch.cat((stimuli_set, stimuli), dim=0)
#     print(stimuli_set.shape)
#     torch.save(stimuli_set, os.path.join(output_dir, f"{roi}_top{topk}_{clip_name}_prompt_extra1.pt"))



