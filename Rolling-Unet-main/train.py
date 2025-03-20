# train.py

import argparse
import os
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import yaml
from albumentations import Compose, RandomRotate90, Flip, Resize, Normalize, RandomBrightnessContrast, ShiftScaleRotate, ElasticTransform, GridDistortion
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs  # 假设 archs.py 定义了模型架构
import losses  # 假设 losses.py 定义了损失函数
from dataset import IAH_Dataset  # 您提供的 dataset.py
from metrics import iou_score, dice_score  # 假设 metrics.py 定义了这些函数
from utils import AverageMeter, str2bool  # 假设 utils.py 定义了这些
import time
# from tensorboardX import SummaryWriter  # 移除 TensorBoard 以减少IO操作
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import cv2  # 确保导入 cv2

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def custom_collate_fn(batch):
    """
    自定义的 collate 函数，确保 filenames 被作为列表返回。
    """
    images, masks, filenames = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks, list(filenames)  # 确保 filenames 是一个列表

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')

    # CUDA settings
    parser.add_argument('--gpu_ids', default='3', type=str, help='Comma separated list of GPU IDs to use (e.g., "0,1")')

    # Model name and training settings
    parser.add_argument('--name', default=None, help='Model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='Mini-batch size (default: 8)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader workers')

    # Model parameters
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Rolling_Unet_L', help='Model architecture')
    parser.add_argument('--deep_supervision', default=False, type=str2bool, help='Use deep supervision')
    parser.add_argument('--input_channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes')
    parser.add_argument('--input_w', default=512, type=int, help='Image width (default: 512)')
    parser.add_argument('--input_h', default=512, type=int, help='Image height (default: 512)')

    # Loss parameters
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES, help='Loss function: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')

    # Data parameters
    parser.add_argument('--split_file', default='train_val_split.txt', help='Path to train_val_split.txt')
    parser.add_argument('--img_dir', default='/zjy/Test/Data/hemat/', help='Root directory for images')
    parser.add_argument('--mask_dir', default='/zjy/Test/Data/hemat/masks/', help='Directory for masks')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', choices=['AdamW', 'SGD'], help='Optimizer: AdamW | SGD (default: AdamW)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum (only for SGD)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='Use Nesterov momentum (only for SGD)')

    # Scheduler parameters
    parser.add_argument('--scheduler', default='WarmupCosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'WarmupCosineAnnealingLR'],
                        help='Learning rate scheduler')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float, help='Factor for ReduceLROnPlateau')
    parser.add_argument('--patience', default=5, type=int, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--milestones', default='50,100,150', type=str, help='Milestones for MultiStepLR')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma for MultiStepLR')
    parser.add_argument('--early_stopping', default=100, type=int, help='Early stopping (default: 100)')
    
    # Gradient accumulation
    parser.add_argument('--accumulation_steps', default=4, type=int, help='Number of gradient accumulation steps')
    
    # Output directory for validation images
    parser.add_argument('--output_dir', default='output/', type=str, help='Directory to save output images during validation')
    
    config = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES
    if config.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    return config

def read_split_file(split_file):
    """
    Reads the train_val_split.txt and returns train and val image paths.
    Expected format:
    Train:
    A/image1.png
    None/image2.png
    ...
    Val:
    A/image3.png
    None/image4.png
    ...
    """
    train_paths = []
    val_paths = []
    current_split = None
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Train:"):
                current_split = 'train'
                continue
            elif line.startswith("Val:"):
                current_split = 'val'
                continue
            else:
                if current_split == 'train':
                    train_paths.append(line)
                elif current_split == 'val':
                    val_paths.append(line)
    return train_paths, val_paths

def train(config, train_loader, model, criterion, optimizer, scaler, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader), desc='Training', leave=False)
    for batch_idx, (input, target, _) in enumerate(train_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 确保 target 有通道维度
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        with autocast():
            # 前向传播
            output = model(input)
            loss = criterion(output, target) / config.accumulation_steps  # 归一化损失

        # 反向传播并缩放梯度
        scaler.scale(loss).backward()

        # 梯度累积
        if (batch_idx + 1) % config.accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 计算指标
        with torch.no_grad():
            output_prob = torch.sigmoid(output)
            output_pred = (output_prob >= 0.5).float()

            iou = iou_score(output_pred, target)
            dice = dice_score(output_pred, target)

        avg_meters['loss'].update(loss.item() * config.accumulation_steps, input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def validate(config, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Validation', leave=False)
        for batch_idx, (input, target, _) in enumerate(val_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Ensure target has channel dimension
            if target.dim() == 3:
                target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            with autocast():
                # Forward pass
                output = model(input)
                loss = criterion(output, target)

            # Apply Sigmoid activation
            output_prob = torch.sigmoid(output)
            # Thresholding
            output_pred = (output_prob >= 0.5).float()

            # Calculate metrics
            iou = iou_score(output_pred, target)
            dice = dice_score(output_pred, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plot_metrics(log, model_name):
    epochs = log['epoch']
    plt.figure(figsize=(18, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, log['loss'], label='Train Loss', color='blue')
    plt.plot(epochs, log['val_loss'], label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Plot IoU
    plt.subplot(1, 3, 2)
    plt.plot(epochs, log['iou'], label='Train IoU', color='blue')
    plt.plot(epochs, log['val_iou'], label='Val IoU', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Curve')
    plt.legend()
    plt.grid(True)

    # Plot Dice
    plt.subplot(1, 3, 3)
    plt.plot(epochs, log['dice'], label='Train Dice', color='blue')
    plt.plot(epochs, log['val_dice'], label='Val Dice', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(f"models/{model_name}/metrics", exist_ok=True)
    plt.savefig(f"models/{model_name}/metrics/metrics_curve.png")
    plt.close()
    print(f"Saved metrics curve at models/{model_name}/metrics/metrics_curve.png")

def main():
    seed_torch()
    config = parse_args()

    # 打印配置以验证 output_dir
    print("Configuration:")
    for key in vars(config):
        print(f"{key}: {getattr(config, key)}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s): {config.gpu_ids}")

    # 修改当前时间格式，避免在 Windows 中使用冒号
    current_time = time.strftime("%Y-%m-%dT%H-%M", time.localtime())  # 将 ':' 替换为 '-'

    if config.name is None:
        if config.deep_supervision:
            config.name = f"{config.arch}_wDS_{current_time}"
        else:
            config.name = f"{config.arch}_woDS_{current_time}"

    os.makedirs(f"models/{config.name}", exist_ok=True)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    print('-' * 20)
    for key in vars(config):
        print(f"{key}: {getattr(config, key)}")
    print('-' * 20)

    with open(f"models/{config.name}/config.yml", 'w') as f:
        yaml.dump(vars(config), f)

    # 定义损失函数 (criterion)
    if config.loss == 'BCEDiceLoss':
        criterion = losses.BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)  # 调整权重为均衡
    elif config.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config.loss]().to(device)

    cudnn.benchmark = True

    # 创建模型
    model = archs.__dict__[config.arch](num_classes=config.num_classes,
                                       input_channels=config.input_channels,
                                       deep_supervision=config.deep_supervision,
                                       img_size=config.input_h)  # 确保 img_size 匹配 input_h

    # 加载预训练权重
    pre_trained_weights = f'path_to_pretrained_weights/{config.arch}_pretrained.pth'
    if os.path.exists(pre_trained_weights):
        try:
            model.load_state_dict(torch.load(pre_trained_weights, map_location=device), strict=False)
            print(f"Loaded pre-trained weights from {pre_trained_weights}")
        except Exception as e:
            print(f"Failed to load pre-trained weights: {e}")
    else:
        print("Pre-trained weights not found, training from scratch.")

    model = model.to(device)
    print(f"Model is moved to device: {device}")

    # 如果使用多 GPU，使用 DistributedDataParallel (更高效)
    if num_gpus > 1:
        print(f"Using DistributedDataParallel with {num_gpus} GPUs")
        torch.distributed.init_process_group(backend='nccl')
        model = nn.parallel.DistributedDataParallel(model)
    else:
        # 使用 DataParallel 作为备选
        if num_gpus > 1:
            print(f"Using DataParallel with {num_gpus} GPUs")
            model = nn.DataParallel(model)

    # 打印总可训练参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # 检查冻结的参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} is frozen.")

    # 设置优化器
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.lr, momentum=config.momentum,
                              nesterov=config.nesterov, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # 设置学习率调度器
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr)
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.factor, patience=config.patience,
                                                   verbose=1, min_lr=config.min_lr)
    elif config.scheduler == 'MultiStepLR':
        milestones = [int(e) for e in config.milestones.split(',')]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                             gamma=config.gamma)
    elif config.scheduler == 'ConstantLR':
        scheduler = None
    elif config.scheduler == 'WarmupCosineAnnealingLR':
        try:
            from warmup_scheduler import GradualWarmupScheduler
        except ImportError:
            raise ImportError("Please install warmup_scheduler: pip install warmup_scheduler")
        base_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=base_scheduler)
    else:
        raise NotImplementedError

    # 读取 train_val_split.txt
    train_paths, val_paths = read_split_file(config.split_file)

    # 定义数据增强
    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.3),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        # ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        # GridDistortion(p=0.3),
        Resize(config.input_h, config.input_w, interpolation=cv2.INTER_LINEAR),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
        ToTensorV2(),
    ])

    val_transform = Compose([
        Resize(config.input_h, config.input_w, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
        ToTensorV2(),
    ])

    # 创建数据集
    train_dataset = IAH_Dataset(
        img_paths=train_paths,
        img_dir=config.img_dir,
        mask_dir=config.mask_dir,
        num_classes=config.num_classes,
        transform=train_transform)
    val_dataset = IAH_Dataset(
        img_paths=val_paths,
        img_dir=config.img_dir,
        mask_dir=config.mask_dir,
        num_classes=config.num_classes,
        transform=val_transform)

    # 创建数据加载器
    # 使用 DistributedSampler 以确保在分布式训练中每个进程加载不同的数据
    if num_gpus > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size * max(num_gpus, 1),  # 根据 GPU 数量调整批量大小
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size * max(num_gpus, 1),
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=custom_collate_fn)

    # 创建混合精度训练的 GradScaler
    scaler = GradScaler()

    # 初始化日志
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config.epochs):
        print(f'Epoch [{epoch+1}/{config.epochs}]')

        if num_gpus > 1:
            train_loader.sampler.set_epoch(epoch)

        # 训练一个 epoch
        train_log = train(config, train_loader, model, criterion, optimizer, scaler, device)
        # 验证
        val_log_epoch = validate(config, val_loader, model, criterion, device)

        # 更新学习率调度器
        if config.scheduler in ['CosineAnnealingLR', 'WarmupCosineAnnealingLR']:
            scheduler.step()
        elif config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_log_epoch['loss'])
        elif config.scheduler == 'MultiStepLR':
            scheduler.step()

        # 打印日志
        print(f"Train Loss: {train_log['loss']:.4f} - Train IoU: {train_log['iou']:.4f} - Train Dice: {train_log['dice']:.4f} "
              f"- Val Loss: {val_log_epoch['loss']:.4f} - Val IoU: {val_log_epoch['iou']:.4f} - Val Dice: {val_log_epoch['dice']:.4f}")

        # 更新日志
        log['epoch'].append(epoch + 1)
        log['lr'].append(config.lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log_epoch['loss'])
        log['val_iou'].append(val_log_epoch['iou'])
        log['val_dice'].append(val_log_epoch['dice'])

        # 提前停止
        if config.early_stopping >= 0:
            if val_log_epoch['iou'] > best_iou:
                torch.save(model.state_dict(), f'models/{config.name}/model.pth')
                best_iou = val_log_epoch['iou']
                print("=> saved best model")
                trigger = 0
            else:
                trigger += 1
                print(f"Early stopping trigger: {trigger}/{config.early_stopping}")
                if trigger >= config.early_stopping:
                    print("=> early stopping")
                    break
        else:
            if val_log_epoch['iou'] > best_iou:
                torch.save(model.state_dict(), f'models/{config.name}/model.pth')
                best_iou = val_log_epoch['iou']
                print("=> saved best model")

        # torch.cuda.empty_cache()

    # 保存日志到 CSV 文件
    pd.DataFrame(log).to_csv(f'models/{config.name}/log.csv', index=False)

    # 绘制并保存指标曲线
    plot_metrics(log, config.name)

    print("Training completed.")

if __name__ == '__main__':
    main()
