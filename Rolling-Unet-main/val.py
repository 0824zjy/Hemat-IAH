import argparse
import os
from collections import OrderedDict
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2  # 正确导入 ToTensorV2
from tqdm import tqdm
import archs
import losses
from metrics import iou_score, dice_score
from utils import AverageMeter
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='Rolling_Unet_L_woDS', help='model name')
    parser.add_argument('--split_file', default='train_val_split.txt', help='Path to train_val_split.txt')
    parser.add_argument('--img_dir', default='/zjy/Test/Data/hemat/', help='Root directory for images')
    parser.add_argument('--mask_dir', default='/zjy/Test/Data/hemat/masks/', help='Directory for masks')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='mini-batch size(default: 4)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader workers (set to 0 for debugging)')
    parser.add_argument('--gpu_ids', default='3', type=str, help='Comma separated list of GPU IDs to use (e.g., "0,1")')

    args = parser.parse_args()
    return args

def read_split_file(split_file):
    """
    Reads the train_val_split.txt and returns train and val image paths.
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

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def unnormalize(tensor, mean, std):
    """
    反归一化图像张量。

    :param tensor: 形状为 [C, H, W] 的张量
    :param mean: 均值列表
    :param std: 标准差列表
    :return: 反归一化后的张量
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)  # 确保值在 [0, 1] 范围内
    return tensor

class CustomIAH_Dataset(Dataset):
    def __init__(self, img_paths, img_dir, mask_dir, num_classes, transform=None):
        """
        初始化数据集。

        :param img_paths: 图像路径列表（相对于 img_dir 的相对路径）
        :param img_dir: 图像的根目录
        :param mask_dir: 掩码的根目录
        :param num_classes: 类别数量
        :param transform: 数据增强和预处理的转换函数
        """
        self.img_paths = img_paths
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 提取图像文件名
        img_name = os.path.basename(img_path)
        # 分离文件名和扩展名
        name, ext = os.path.splitext(img_name)
        # 构建掩码文件名
        mask_name = f"{name}_mask{ext}"
        # 构建掩码路径
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 加载图像
        image = cv2.imread(os.path.join(self.img_dir, img_path))
        if image is None:
            raise ValueError(f"Failed to read image: {os.path.join(self.img_dir, img_path)}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            # 如果掩码文件不存在，创建一个全黑掩码
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            # 可选：记录缺失掩码的图像路径，便于后续检查
            print(f"Warning: Mask not found for {img_path}. Using a black mask.")
        else:
            # 加载掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # 如果掩码文件存在但加载失败，也创建全黑掩码
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                print(f"Warning: Failed to read mask: {mask_path}. Using a black mask.")
            else:
                mask = mask / 255.0  # 归一化到 [0, 1]

        # 应用转换（如数据增强）
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 由于 ToTensorV2 已经将 image 和 mask 转换为 Tensor，无需再次转换
        if self.num_classes > 1:
            mask = torch.nn.functional.one_hot(mask.long(), num_classes=self.num_classes).permute(2, 0, 1).float()
        else:
            mask = mask.unsqueeze(0).float()  # 确保掩码维度为 [1, H, W]

        return image, mask, {'img_path': img_path}

def visualize_predictions(model, dataloader, save_dir='inference_results', device='cuda', image_paths=None):
    """
    可视化模型的预测结果，并将结果保存到指定目录。

    :param model: 已加载权重的模型
    :param dataloader: 数据加载器
    :param save_dir: 保存结果的目录
    :param device: 设备 ('cpu' 或 'cuda')
    :param image_paths: 图像路径列表，用于获取图像名称
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    
    # 定义反归一化所需的均值和标准差
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    with torch.no_grad():
        for batch_idx, (images, masks, meta) in enumerate(tqdm(dataloader, desc='Visualizing Predictions')):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # 二值化预测

            preds = preds.cpu().numpy()
            images = images.cpu()
            masks = masks.cpu().numpy()
            img_paths = meta['img_path']

            for i in range(images.size(0)):
                img = images[i]
                try:
                    mask = masks[i].squeeze(0)  # [H, W]
                except ValueError as e:
                    print(f"Error squeezing mask: {e}")
                    print(f"Mask shape: {masks[i].shape}")
                    continue  # 跳过此样本

                pred = preds[i].squeeze(0)  # [H, W]

                # 反归一化图像
                img_unnormalized = unnormalize(img, mean, std)

                # 转换为 PIL 图像
                original_img = T.ToPILImage()(img_unnormalized)
                true_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
                pred_mask = Image.fromarray((pred * 255).astype(np.uint8)).convert("L")

                # 创建彩色掩码
                true_mask_color = Image.new("RGBA", original_img.size, (0, 0, 255, 0))  # 绿色
                pred_mask_color = Image.new("RGBA", original_img.size, (255, 0, 0, 0))  # 红色

                # 将掩码转换为透明度图
                true_mask_alpha = true_mask.point(lambda p: p > 0 and 100)  # 半透明
                pred_mask_alpha = pred_mask.point(lambda p: p > 0 and 150)  # 更不透明

                # 将透明度应用到颜色图
                true_mask_color.putalpha(true_mask_alpha)
                pred_mask_color.putalpha(pred_mask_alpha)

                # 叠加掩码到原图
                overlay = original_img.convert("RGBA")
                overlay = Image.alpha_composite(overlay, true_mask_color)
                overlay = Image.alpha_composite(overlay, pred_mask_color)

                # 创建组合图像
                combined_width = original_img.width * 4
                combined_height = original_img.height
                combined = Image.new('RGB', (combined_width, combined_height))

                combined.paste(original_img, (0, 0))
                combined.paste(true_mask.convert("RGB"), (original_img.width, 0))
                combined.paste(pred_mask.convert("RGB"), (original_img.width * 2, 0))
                combined.paste(overlay.convert("RGB"), (original_img.width * 3, 0))

                # 添加标题
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()
                draw = ImageDraw.Draw(combined)
                draw.text((10, 5), "Original | True Mask | Predicted Mask | Overlay", fill=(255, 255, 255), font=font)

                # 获取图像名称
                img_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
                save_name = f"{img_name}_test.png"
                combined.save(os.path.join(save_dir, save_name))

    print(f"Inference results saved to {save_dir}")

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # 切换到评估模式
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Validation')
        for batch_idx, (input, target, meta) in enumerate(val_loader):
            input = input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

            # 确保目标具有通道维度
            if target.dim() == 3:
                target = target.unsqueeze(1)  # 将 [B, H, W] 转换为 [B, 1, H, W]

            with autocast():
                # 计算输出
                output = model(input)
                loss = criterion(output, target)
                # 应用 Sigmoid 激活
                output_prob = torch.sigmoid(output)
                # 阈值化
                output_pred = (output_prob >= 0.5).float()

            # 计算指标
            iou = iou_score(output_pred, target)
            dice = dice_score(output_pred, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', f"{avg_meters['loss'].avg:.4f}"),
                ('iou', f"{avg_meters['iou'].avg:.4f}"),
                ('dice', f"{avg_meters['dice'].avg:.4f}")
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main():
    seed_torch()
    args = parse_args()

    # 设置 CUDA 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s): {args.gpu_ids}")

    # 读取配置文件
    config_path = f"models/{args.name}/config.yml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print(f"{key}: {str(config[key])}")
    print('-' * 20)

    cudnn.benchmark = True

    print(f"=> creating model {config['arch']}")
    model = archs.__dict__[config['arch']](num_classes=config['num_classes'],
                                           input_channels=config['input_channels'],
                                           deep_supervision=config.get('deep_supervision', False),
                                           pretrained=True)  # 使用预训练模型（如果支持）

    model = model.to(device)

    # 如果使用多个 GPU，使用 DataParallel
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    # 定义损失函数（criterion）
    if config['loss'] == 'BCEDiceLoss':
        criterion = losses.BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)  # 调整权重
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    # 读取 train_val_split.txt
    train_paths, val_paths = read_split_file(args.split_file)

    # 定义验证转换（使用 ToTensorV2）
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # 使用 ToTensorV2
    ])

    # 创建验证数据集和数据加载器，使用 CustomIAH_Dataset
    val_dataset = CustomIAH_Dataset(
        img_paths=val_paths,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * num_gpus,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True)

    # 加载模型权重
    model_path = f"models/{args.name}/model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    state_dict = torch.load(model_path, map_location=device)

    # 判断 state_dict 是否包含 'module.' 前缀
    module_keys = any(k.startswith('module.') for k in state_dict.keys())
    model_is_dataparallel = isinstance(model, torch.nn.DataParallel)

    if module_keys and not model_is_dataparallel:
        # 如果 state_dict 有 'module.' 前缀但当前模型未使用 DataParallel，移除 'module.'
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    elif not module_keys and model_is_dataparallel:
        # 如果 state_dict 没有 'module.' 前缀但当前模型使用了 DataParallel，添加 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = 'module.' + k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        # 其他情况直接加载
        model.load_state_dict(state_dict)

    model.eval()
    print("Model loaded successfully.")

    # 初始化指标
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # 可视化预测结果的保存目录
    visualize_save_dir = 'inference_results_unet'
    if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

    # 进行验证并计算指标
    val_log = validate(config, val_loader, model, criterion)

    print(f'Validation Loss: {val_log["loss"]:.4f}')
    print(f'Validation IoU: {val_log["iou"]:.4f}')
    print(f'Validation Dice: {val_log["dice"]:.4f}')

    # 可视化预测结果
    print("Starting visualization of predictions...")
    visualize_predictions(model, val_loader, save_dir=visualize_save_dir, device=device, image_paths=val_paths)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
