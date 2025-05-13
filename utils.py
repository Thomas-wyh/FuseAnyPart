import os
import argparse
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--img_id_dict_json",
        type=str,
        default=None,
        required=True,
        help="Path to image-id dict",
    )
    parser.add_argument(
        "--id_img_dict_json",
        type=str,
        default="",
        required=True,
        help="Path to id-image dict.",
    )
    parser.add_argument(
        "--ref_image_root_path",
        type=str,
        default="",
        required=True,
        help="Path to reference images.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--train_unet", action="store_true", help="train unet or not"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        type=bool, 
        default=True,
        help="weather enable xformers memory efficient attention"
    )
    parser.add_argument(
        "--test_img_json",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--test_save_dir",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

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


def prepare_mask(ref_bboxes, drop_image_embed):
    mask = torch.zeros(ref_bboxes.shape[0], 4, 64, 64)
    ref_bboxes = (64 * ref_bboxes).type(torch.int16)
    for i in range(ref_bboxes.shape[0]):
        if drop_image_embed[i][0] == 1:
            continue
        for j in range(3):
            if drop_image_embed[i][j+1] == 0:
                ref_bboxes[i][j][2] = torch.max(ref_bboxes[i][j][2], ref_bboxes[i][j][0]+1)
                ref_bboxes[i][j][3] = torch.max(ref_bboxes[i][j][3], ref_bboxes[i][j][1]+1)
                mask[i, :, ref_bboxes[i][j][0]:ref_bboxes[i][j][2], ref_bboxes[i][j][1]:ref_bboxes[i][j][3]] = 1
    return mask