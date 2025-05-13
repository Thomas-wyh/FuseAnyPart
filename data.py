import os
import json
import random

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    ref_images = torch.stack([example["ref_image"] for example in data])
    clip_images_face = torch.cat([example["clip_image_face"] for example in data], dim=0)
    clip_images_eyes = torch.cat([example["clip_image_eyes"] for example in data], dim=0)
    clip_images_nose = torch.cat([example["clip_image_nose"] for example in data], dim=0)
    clip_images_mouth = torch.cat([example["clip_image_mouth"] for example in data], dim=0)
    drop_image_embeds = torch.stack([example["drop_image_embed"] for example in data])
    pixel_mask = torch.stack([example["pixel_mask"] for example in data])
    ref_bboxes = torch.stack([example["ref_bbox"] for example in data])
    return {
        "images": images,
        "ref_images": ref_images,
        "clip_images_face": clip_images_face,
        "clip_images_eyes": clip_images_eyes,
        "clip_images_nose": clip_images_nose,
        "clip_images_mouth": clip_images_mouth,
        "drop_image_embeds": drop_image_embeds,
        "pixel_mask": pixel_mask,
        "ref_bboxes": ref_bboxes,
    }

def prepare_pixel_mask(ref_bboxe):
    mask = torch.zeros(1024, 1024)
    ref_bboxe = (1024 * ref_bboxe).type(torch.int16)
    for i in range(3):
        mask[ref_bboxe[i][0]:ref_bboxe[i][2], ref_bboxe[i][1]:ref_bboxe[i][3]] = 1
    return mask

class CelebAHQDataset(torch.utils.data.Dataset):
    def __init__(self, img_id_dict_json, id_img_dict_json, tokenizer, image_root_path, ref_image_root_path, size=512, i_drop_rate=0.04):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.image_root_path = image_root_path
        self.ref_image_root_path = ref_image_root_path
        self.ref_bbox_root_path = ref_image_root_path
        self.img_id_dict = json.load(open(img_id_dict_json))
        self.id_img_dict = json.load(open(id_img_dict_json))
        self.ids = list(self.id_img_dict.keys())

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_files = self.id_img_dict[image_id]
        image_num = len(image_files)
        if image_num == 1:
            tar_image = image_files[0]
            ref_image_eyes = image_files[0]
            ref_image_nose = image_files[0]
            ref_image_mouth = image_files[0]
        else :
            tar_image = random.choice(image_files)
            image_files.remove(tar_image)
            ref_image_eyes, ref_image_nose, ref_image_mouth = np.random.choice(image_files, size=3, replace=True)
            
        ref_bbox = []
        ref_bbox_dict = torch.load(os.path.join(self.ref_bbox_root_path, f"{image_id}/{tar_image}_bbox.pth"))
        ref_bbox_dict_eyes = torch.load(os.path.join(self.ref_bbox_root_path, f"{image_id}/{ref_image_eyes}_bbox.pth"))
        ref_bbox_dict_nose = torch.load(os.path.join(self.ref_bbox_root_path, f"{image_id}/{ref_image_nose}_bbox.pth"))
        ref_bbox_dict_mouth = torch.load(os.path.join(self.ref_bbox_root_path, f"{image_id}/{ref_image_mouth}_bbox.pth"))
        for key in ["eyes", "nose", "mouth"]:
            ref_bbox.append(torch.tensor(ref_bbox_dict[key]))
        ref_bbox.append(torch.tensor(ref_bbox_dict_eyes["eyes"]))
        ref_bbox.append(torch.tensor(ref_bbox_dict_nose["nose"]))
        ref_bbox.append(torch.tensor(ref_bbox_dict_mouth["mouth"]))
        ref_bbox = torch.stack(ref_bbox)
        raw_tar_image = Image.open(os.path.join(self.image_root_path, f"{image_id}/{tar_image}.jpg"))
        raw_ref_image = Image.open(os.path.join(self.ref_image_root_path, f"{image_id}/{tar_image}_masked_face.jpg"))
        image = self.transform(raw_tar_image.convert("RGB"))
        ref_image = self.transform(raw_ref_image.convert("RGB"))

        raw_ref_image_face = Image.open(os.path.join(self.ref_image_root_path, f"{image_id}/{tar_image}_masked_face.jpg"))
        raw_ref_image_eyes = Image.open(os.path.join(self.ref_image_root_path, f"{image_id}/{ref_image_eyes}_masked_eyes.jpg"))
        raw_ref_image_nose = Image.open(os.path.join(self.ref_image_root_path, f"{image_id}/{ref_image_nose}_masked_nose.jpg"))
        raw_ref_image_mouth = Image.open(os.path.join(self.ref_image_root_path, f"{image_id}/{ref_image_mouth}_masked_mouth.jpg"))
        clip_image_face = self.clip_image_processor(images=raw_ref_image_face, return_tensors="pt").pixel_values
        clip_image_eyes = self.clip_image_processor(images=raw_ref_image_eyes, return_tensors="pt").pixel_values
        clip_image_nose = self.clip_image_processor(images=raw_ref_image_nose, return_tensors="pt").pixel_values
        clip_image_mouth = self.clip_image_processor(images=raw_ref_image_mouth, return_tensors="pt").pixel_values

        drop_image_embed = torch.zeros(4)
        rand_num = random.random()
        if rand_num <  self.i_drop_rate:
            drop_image_embed[0] = 1
            drop_image_embed[1] = 1
            drop_image_embed[2] = 1
            drop_image_embed[3] = 1
        elif rand_num <  2 * self.i_drop_rate :
            drop_image_embed[1] = 1
            drop_image_embed[2] = 1
            drop_image_embed[3] = 1
        elif rand_num <  3 * self.i_drop_rate :
            drop_image_embed[2] = 1
            drop_image_embed[3] = 1
        elif rand_num < 4 * self.i_drop_rate :
            drop_image_embed[1] = 1
            drop_image_embed[3] = 1
        elif rand_num < 5 * self.i_drop_rate :
            drop_image_embed[1] = 1
            drop_image_embed[2] = 1
        
        pixel_mask = prepare_pixel_mask(ref_bbox)

        return {
            "image": image,
            "ref_image": ref_image,
            "clip_image_face": clip_image_face,
            "clip_image_eyes": clip_image_eyes,
            "clip_image_nose": clip_image_nose,
            "clip_image_mouth": clip_image_mouth,
            "drop_image_embed": drop_image_embed,
            "pixel_mask": pixel_mask,
            "ref_bbox": ref_bbox
        }

    def __len__(self):
        return len(self.ids)
