python train.py --split_file=/zjy/Test/IAHtrain_val_split.txt --img_dir=/zjy/Test/ --mask_dir=/zjy/Test/Data/IAH/masks/ --arch=Rolling_Unet_L --num_classes=1 --epochs=10 --batch_size=4 --num_workers=4

CUDA_VISIBLE_DEVICES=3 python train.py \
    --name=Rolling_Unet_L_wDS_2024-12-28T12:00 \
    --epochs=200 \
    --batch_size=4 \
    --num_workers=4 \
    --arch=Rolling_Unet_L \
    --deep_supervision=True \
    --input_channels=3 \
    --num_classes=1 \
    --input_w=512 \
    --input_h=512 \
    --loss=BCEDiceLoss \
    --split_file=/zjy/Test/IAHtrain_val_split.txt \
    --img_dir=/zjy/Test/ \
    --mask_dir=/zjy/Test/ \
    --optimizer=AdamW \
    --lr=1e-3 \
    --momentum=0.9 \
    --weight_decay=1e-4 \
    --nesterov=False \
    --scheduler=WarmupCosineAnnealingLR \
    --min_lr=1e-6 \
    --factor=0.1 \
    --patience=5 \
    --milestones=50,100,150 \
    --gamma=0.1 \
    --early_stopping=100 \
    --accumulation_steps=8


python val.py --name=Rolling_Unet_L_woDS --split_file=/zjy/Test/Rolling-Unet-main/train_val_split.txt --num_classes=1 --batch_size=8 --num_workers=8
