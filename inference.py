import os
import random
import json
from tqdm import tqdm
from PIL import Image
from typing import List
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from pipe.pipleline_stable_diffusion import StableDiffusionPipeline
from fuseanypart.resampler import Resampler
from fuseanypart.attention_processor import XFormersAttnProcessor as AttnProcessor
from fuseanypart.attention_processor import XFormersMultirefAttnProcessor_only_add_cross as FAPAttnProcessor######


class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class FuseAnyPart:
    def __init__(self, sd_pipe, image_encoder_path, fap_ckpt, device, num_tokens=16):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.fap_ckpt = fap_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model, self.image_proj_model_2 = self.init_proj()

        self.load_adapter()

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        image_proj_model_2 = MLPProjModel(self.pipe.unet.config.cross_attention_dim, self.image_encoder.config.hidden_size).to(self.device, dtype=torch.float16)
        return image_proj_model, image_proj_model_2

    def set_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = FAPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,   ########
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_adapter(self):
        if os.path.splitext(self.fap_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.fap_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.fap_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.image_proj_model_2.load_state_dict(state_dict["image_proj2"])
        self.pipe.unet.load_state_dict(state_dict["unet"])


    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        image_feature_embeds = self.image_proj_model_2(clip_image_embeds[:, 1:, :])
        image_prompt_embeds = torch.cat([image_prompt_embeds, image_feature_embeds], dim=1)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros(image_prompt_embeds.shape[0], 3, 224, 224).to(self.device, dtype=torch.float16), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        uncond_image_feature_embeds = self.image_proj_model_2(uncond_clip_image_embeds[:, 1:, :])
        uncond_image_prompt_embeds = torch.cat([uncond_image_prompt_embeds, uncond_image_feature_embeds], dim=1)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, FAPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        pixel_values_ref_img=None,
        pixel_mask=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        with torch.no_grad():
            latents_ref_img = self.pipe.vae.encode(pixel_values_ref_img).latent_dist.sample()
            latents_ref_img = latents_ref_img * self.pipe.vae.config.scaling_factor
        latents_ref_img = latents_ref_img.repeat(num_samples, 1, 1, 1)
        pixel_mask = F.interpolate(pixel_mask.unsqueeze(1), size=(latents_ref_img.shape[-2], latents_ref_img.shape[-1]), mode='nearest').repeat(num_samples, 1, 1, 1).to(device=latents_ref_img.device, dtype=latents_ref_img.dtype)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents_ref_img=latents_ref_img,
            pixel_mask=pixel_mask,
            **kwargs,
        ).images

        return images


def prepare_pixel_mask(ref_bboxe):
    mask = torch.zeros(1024, 1024)
    ref_bboxe = (1024 * ref_bboxe).type(torch.int16)
    for i in range(ref_bboxe.shape[0]):
        mask[ref_bboxe[i][0]:ref_bboxe[i][2], ref_bboxe[i][1]:ref_bboxe[i][3]] = 1
    return mask


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def concate_embeds(image_embeds_face, image_embeds_eyes, image_embeds_nose, image_embeds_mouth, ref_bboxes):
    ref_bboxes = (16 * ref_bboxes).type(torch.int16)
    bsz, _, c = image_embeds_face.shape
    face = image_embeds_face[:, 1:, :].clone().permute(0, 2, 1).reshape(bsz, c, 16, 16)
    eyes = image_embeds_eyes[:, 1:, :].permute(0, 2, 1).reshape(bsz, c, 16, 16)
    nose = image_embeds_nose[:, 1:, :].permute(0, 2, 1).reshape(bsz, c, 16, 16)
    mouth = image_embeds_mouth[:, 1:, :].permute(0, 2, 1).reshape(bsz, c, 16, 16)
    for i in range(ref_bboxes.shape[0]):
        for j in range(6):
            ref_bboxes[i][j][2] = torch.max(ref_bboxes[i][j][2], ref_bboxes[i][j][0]+1)
            ref_bboxes[i][j][3] = torch.max(ref_bboxes[i][j][3], ref_bboxes[i][j][1]+1)
        eyes_ = F.interpolate(eyes[i, :, ref_bboxes[i][3][0]:ref_bboxes[i][3][2], ref_bboxes[i][3][1]:ref_bboxes[i][3][3]].unsqueeze(0), (ref_bboxes[i][0][2]-ref_bboxes[i][0][0], ref_bboxes[i][0][3]-ref_bboxes[i][0][1]))
        nose_ = F.interpolate(nose[i, :, ref_bboxes[i][4][0]:ref_bboxes[i][4][2], ref_bboxes[i][4][1]:ref_bboxes[i][4][3]].unsqueeze(0), (ref_bboxes[i][1][2]-ref_bboxes[i][1][0], ref_bboxes[i][1][3]-ref_bboxes[i][1][1]))
        mouth_ = F.interpolate(mouth[i, :, ref_bboxes[i][5][0]:ref_bboxes[i][5][2], ref_bboxes[i][5][1]:ref_bboxes[i][5][3]].unsqueeze(0), (ref_bboxes[i][2][2]-ref_bboxes[i][2][0], ref_bboxes[i][2][3]-ref_bboxes[i][2][1]))
        face[i, :, ref_bboxes[i][0][0]:ref_bboxes[i][0][2], ref_bboxes[i][0][1]:ref_bboxes[i][0][3]] = eyes_[0]
        face[i, :, ref_bboxes[i][1][0]:ref_bboxes[i][1][2], ref_bboxes[i][1][1]:ref_bboxes[i][1][3]] = nose_[0]
        face[i, :, ref_bboxes[i][2][0]:ref_bboxes[i][2][2], ref_bboxes[i][2][1]:ref_bboxes[i][2][3]] = mouth_[0]
    face = face.reshape(bsz, c, -1).permute(0, 2, 1)
    image_embeds_face[:, 1:, :] = face
    return image_embeds_face


def get_clip_image_embeds(ip_model, device, weight_dtype, a, b, c, d, data_root_path, ref_image_root_path):
    ref_bbox_dict = torch.load(os.path.join(ref_image_root_path, f"{a}_bbox.pth"))
    ref_bbox_dict_eyes = torch.load(os.path.join(ref_image_root_path, f"{b}_bbox.pth"))
    ref_bbox_dict_nose = torch.load(os.path.join(ref_image_root_path, f"{c}_bbox.pth"))
    ref_bbox_dict_mouth = torch.load(os.path.join(ref_image_root_path, f"{d}_bbox.pth"))
    ref_bbox = []
    for key in ["eyes", "nose", "mouth"]:
        ref_bbox.append(torch.tensor(ref_bbox_dict[key]))
    ref_bbox.append(torch.tensor(ref_bbox_dict_eyes['eyes']))
    ref_bbox.append(torch.tensor(ref_bbox_dict_nose['nose']))
    ref_bbox.append(torch.tensor(ref_bbox_dict_mouth['mouth']))
    ref_bboxes = torch.stack(ref_bbox).unsqueeze(0)

    image_face = Image.open(f"{ref_image_root_path}/{a}_masked_face.jpg").resize((256, 256))
    image_eyes = Image.open(f"{ref_image_root_path}/{b}_masked_eyes.jpg").resize((256, 256))
    image_nose = Image.open(f"{ref_image_root_path}/{c}_masked_nose.jpg").resize((256, 256))
    image_mouth = Image.open(f"{ref_image_root_path}/{d}_masked_mouth.jpg").resize((256, 256))
    # get image embeds
    face_clip_image = ip_model.clip_image_processor(images=image_face, return_tensors="pt").pixel_values
    face_clip_image_embeds = ip_model.image_encoder(face_clip_image.to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
    eyes_clip_image = ip_model.clip_image_processor(images=image_eyes, return_tensors="pt").pixel_values
    eyes_clip_image_embeds = ip_model.image_encoder(eyes_clip_image.to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
    nose_clip_image = ip_model.clip_image_processor(images=image_nose, return_tensors="pt").pixel_values
    nose_clip_image_embeds = ip_model.image_encoder(nose_clip_image.to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
    mouth_clip_image = ip_model.clip_image_processor(images=image_mouth, return_tensors="pt").pixel_values
    mouth_clip_image_embeds = ip_model.image_encoder(mouth_clip_image.to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
    return concate_embeds(face_clip_image_embeds, eyes_clip_image_embeds, nose_clip_image_embeds, mouth_clip_image_embeds, ref_bboxes)


if __name__ == "__main__":
    pretrained_model_name_or_path = "pretrained_weight/Realistic_Vision_V4.0_noVAE"
    image_encoder_path = "pretrained_weight/image_encoder"
    fap_ckpt = "pretrained_weight/fap.bin"
    device = "cuda"
    
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ])
    weight_dtype = torch.float16  
    
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    in_channels = 9
    unet.register_to_config(in_channels=in_channels)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            in_channels, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias.copy_(unet.conv_in.bias)
        unet.conv_in = new_conv_in
    
    unet = unet.to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device, dtype=weight_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device, dtype=weight_dtype)
    
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    fuseanypart = FuseAnyPart(pipe, image_encoder_path, fap_ckpt, device, num_tokens=16)
    
    image_root_path = "assets/test/images"
    ref_image_root_path = "assets/test/croped_images"
    ori_image_root_path = 'assets/test/ori_images'
    
    face_img = "musk"
    eyes_img = "zuck"
    nose_img = "sam"
    mouth_img = "kobe"
    
    clip_image_embeds = get_clip_image_embeds(fuseanypart, device, weight_dtype, face_img, eyes_img, nose_img, mouth_img, image_root_path, ref_image_root_path)
    raw_ref_image = Image.open(os.path.join(ref_image_root_path, f"{face_img}_masked_face.jpg"))
    pixel_values_ref_img = transform(raw_ref_image.convert("RGB")).unsqueeze(0).to("cuda", dtype=weight_dtype)
    
    ref_bbox = []
    ref_bbox_dict = torch.load(os.path.join(ref_image_root_path, f"{face_img}_bbox.pth"))
    ref_bbox_dict_eyes = torch.load(os.path.join(ref_image_root_path, f"{face_img}_bbox.pth"))
    ref_bbox_dict_nose = torch.load(os.path.join(ref_image_root_path, f"{face_img}_bbox.pth"))
    ref_bbox_dict_mouth = torch.load(os.path.join(ref_image_root_path, f"{face_img}_bbox.pth"))
    for key in ["eyes", "nose", "mouth"]:
        ref_bbox.append(torch.tensor(ref_bbox_dict[key]))
    ref_bbox.append(torch.tensor(ref_bbox_dict_eyes["eyes"]))
    ref_bbox.append(torch.tensor(ref_bbox_dict_nose["nose"]))
    ref_bbox.append(torch.tensor(ref_bbox_dict_mouth["mouth"]))
    ref_bbox = torch.stack(ref_bbox)
    pixel_mask = prepare_pixel_mask(ref_bbox).unsqueeze(0).to("cuda")
    images = []
    images.append(Image.open(f"{ori_image_root_path}/{face_img}.png").resize((512, 512)))
    images.append(Image.open(f"{ori_image_root_path}/{eyes_img}.png").resize((512, 512)))
    images.append(Image.open(f"{ori_image_root_path}/{nose_img}.png").resize((512, 512)))
    images.append(Image.open(f"{ori_image_root_path}/{mouth_img}.png").resize((512, 512)))
    images += fuseanypart.generate(clip_image_embeds = clip_image_embeds, prompt=None, num_samples=1, num_inference_steps=50, seed=-1, scale=1, guidance_scale=7.5, pixel_values_ref_img=pixel_values_ref_img, pixel_mask=pixel_mask)
    grid = image_grid(images, 1, 5)
    grid.save("assets/test/result.png")
