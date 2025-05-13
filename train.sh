
accelerate launch train.py --pretrained_model_name_or_path ./pretrained_weight/Realistic_Vision_V4.0_noVAE \
        --image_encoder_path ./pretrained_weight/image_encoder \
        --output_dir ./output/ckpt \
        --image_root_path ./datasets/celebAHQ/img_id \
        --ref_image_root_path ./datasets/celebAHQ/img_id \
        --img_id_dict_json ./datasets/celebAHQ/img_id/img_id_dict.json \
        --id_img_dict_json ./datasets/celebAHQ/img_id/train_id_img.json \
        --logging_dir ./output/ckpt/log \
        --train_batch_size 1 \
        --save_steps 1000 \
        --train_unet
