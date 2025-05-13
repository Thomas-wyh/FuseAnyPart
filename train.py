import os
import random
from pathlib import Path
from einops import rearrange
import json
import itertools
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDPMScheduler
from models.unet import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPVisionModelWithProjection

from utils import parse_args, concate_embeds
from data import CelebAHQDataset, collate_fn
from fuseanypart.resampler import Resampler
from fuseanypart.attention_processor import XFormersAttnProcessor as AttnProcessor
from fuseanypart.attention_processor import XFormersMultirefAttnProcessor as FAPAttnProcessor



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


class Adapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, image_feature=image_embeds[:, 1:, :]).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        self.adapter_modules.load_state_dict(state_dict["adapter"], strict=True)

        # Calculate new checksums
        new_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_proj_sum != new_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")



def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

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

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    image_proj_model = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            attn_procs[name] = FAPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.num_tokens)
    unet.set_attn_processor(attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    adapter = Adapter(unet, image_proj_model, adapter_modules, args.pretrained_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
        
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)


    # optimizer
    if args.train_unet:
        adapter.unet.requires_grad_(True)
        params_to_opt = itertools.chain(adapter.image_proj_model.parameters(), adapter.unet.parameters())
    else:
        params_to_opt = itertools.chain(adapter.image_proj_model.parameters(), adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = CelebAHQDataset(args.img_id_dict_json, args.id_img_dict_json , tokenizer=tokenizer, size=args.resolution, image_root_path=args.image_root_path, ref_image_root_path=args.ref_image_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    adapter, optimizer, train_dataloader = accelerator.prepare(adapter, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Convert ref images to latent space 
                    latents_ref_img = vae.encode(batch["ref_images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents_ref_img = latents_ref_img * vae.config.scaling_factor

                pixel_mask = batch["pixel_mask"].to(device=accelerator.device)
                pixel_mask = F.interpolate(pixel_mask.unsqueeze(1), size=(latents_ref_img.shape[-2], latents_ref_img.shape[-1]), mode='nearest') # shape is b c w h
                latents_ref_img = latents_ref_img * (1 - pixel_mask) + torch.randn_like(latents_ref_img) * pixel_mask


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = torch.cat([noisy_latents, latents_ref_img, pixel_mask], dim=1)
                clip_image_face_ = []
                clip_image_eyes_ = []
                clip_image_nose_ = []
                clip_image_mouth_ = []
                for clip_image_face, clip_image_eyes, clip_image_nose, clip_image_mouth, drop_image_embed in zip(batch["clip_images_face"], batch["clip_images_eyes"], batch["clip_images_nose"], batch["clip_images_mouth"], batch["drop_image_embeds"]):
                    clip_image_face_.append(torch.zeros_like(clip_image_face) if drop_image_embed[0] == 1 else clip_image_face)
                    clip_image_eyes_.append(torch.zeros_like(clip_image_eyes) if drop_image_embed[1] == 1 else clip_image_eyes) 
                    clip_image_nose_.append(torch.zeros_like(clip_image_nose) if drop_image_embed[2] == 1 else clip_image_nose)
                    clip_image_mouth_.append(torch.zeros_like(clip_image_mouth) if drop_image_embed[3] == 1 else clip_image_mouth) 
                clip_image_face_ = torch.stack(clip_image_face_, dim=0)
                clip_image_eyes_ = torch.stack(clip_image_eyes_, dim=0)
                clip_image_nose_ = torch.stack(clip_image_nose_, dim=0)
                clip_image_mouth_ = torch.stack(clip_image_mouth_, dim=0)
                with torch.no_grad():
                    image_embeds_face = image_encoder(clip_image_face_.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    image_embeds_eyes = image_encoder(clip_image_eyes_.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    image_embeds_nose = image_encoder(clip_image_nose_.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    image_embeds_mouth = image_encoder(clip_image_mouth_.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]

                    ref_bboxes = batch["ref_bboxes"]
                    image_embeds = concate_embeds(image_embeds_face, image_embeds_eyes, image_embeds_nose, image_embeds_mouth, ref_bboxes)

                with torch.no_grad():
                    encoder_hidden_states = torch.zeros(bsz, 77, 768).to(accelerator.device)

                noise_pred = adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
                    
            global_step += 1
            
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                accelerator.print(f"save checkpoint-{global_step}")
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
