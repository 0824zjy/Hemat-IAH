可视化服务，可以启动也可以不用
python -m visdom.server

训练：
python scripts/segmentation_train.py \
    --data_dir ./data/training \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --batch_size 4 \
    --microbatch -1 \
    --ema_rate 0.9999 \
    --log_interval 50 \
    --save_interval 10 \
    --use_fp16 False \
    --fp16_scale_growth 1e-3 \
    --lr_anneal_steps 200 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --image_size 64 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --dropout 0.0 \
    --channel_mult 1,2,3,4

python scripts/segmentation_train.py \
    --data_dir ./data/training \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --batch_size 4 \
    --microbatch -1 \
    --ema_rate 0.9999 \
    --log_interval 50 \
    --save_interval 10 \
    --use_fp16 False \
    --fp16_scale_growth 1e-3 \
    --lr_anneal_steps 1000 \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --image_size 128 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --num_heads 4 \
    --attention_resolutions 16,8 \
    --dropout 0.1 \
    --channel_mult 1,2,3,4 \
    --num_epochs 200  \
    --gpu 0

测试：CUDA_LAUNCH_BLOCKING=3 python scripts/segmentation_sample.py
CUDA_LAUNCH_BLOCKING=0 python scripts/segmentation_sample.py     --model_path /zjy/Test/results/best_model.pt     --data_split /zjy/Test/val_split.txt     --output_dir /zjy/Test/results/inference     --image_size 64     --num_channels 128     --num_res_blocks 2     --num_heads 4     --attention_resolutions 16,8     --channel_mult 1,2,3,4     --dropout 0.0     --diffusion_steps 1000     --noise_schedule linear     --use_fp16 False     --batch_size 1 


