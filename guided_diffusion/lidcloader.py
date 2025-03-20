import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import os
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

class HematDataset(Dataset):
    def __init__(self, train_split_file, val_split_file, test_flag=True, target_size=(256, 256), filter_paths=None):
        """
        初始化数据集类
        :param train_split_file: 训练集路径分割文件
        :param val_split_file: 验证集路径分割文件
        :param test_flag: 是否为测试模式（测试/验证时不做数据增强）
        :param target_size: 图像和掩码的固定尺寸
        :param filter_paths: 过滤路径，仅加载特定图像路径（如验证集）
        """
        super().__init__()
        self.test_flag = test_flag
        self.database = []
        self.target_size = target_size  # 固定图像大小

        data_split_file = val_split_file if test_flag else train_split_file

        with open(data_split_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.startswith(("Train:", "Val:")):
                if filter_paths:
                    if line in filter_paths:
                        self.database.append(line)
                else:
                    self.database.append(line)

        self.transform_image = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def augment(self, image, mask):
        """
        对图像和掩码同时做随机增强（水平翻转、垂直翻转和旋转）
        :param image: PIL 格式的图像
        :param mask:  PIL 格式的掩码
        :return: 增强后的 image 和 mask
        """
        if np.random.rand() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if np.random.rand() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        angle = np.random.uniform(-10, 10)
        image = F.rotate(image, angle, resample=Image.BILINEAR)
        mask = F.rotate(mask, angle, resample=Image.NEAREST)
        return image, mask

    def __getitem__(self, index):
        """
        获取数据集中的单个样本
        :param index: 数据索引
        :return: 图像、掩码、文件名
        """
        image_path = self.database[index]
        mask_path = image_path.replace("Data/IAH/A/", "Data/IAH/masks/").replace(".bmp", "_mask.bmp")
        image_np = io.imread(image_path)  # [H, W, 3] 或 [H, W]
        if image_np.ndim == 2:  # 灰度图扩展为 3 通道
            image_np = np.expand_dims(image_np, axis=-1).repeat(3, axis=-1)
        image_pil = Image.fromarray(image_np)

        if "None" in image_path:  # 负样本，构造全零掩码
            mask_pil = Image.fromarray(np.zeros(self.target_size, dtype=np.uint8))
        else:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask_np = io.imread(mask_path)  # [H, W] 或 [H, W, 3]
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]  # 如果掩码为 3 通道，取第一个通道
            mask_pil = Image.fromarray(mask_np)
        if not self.test_flag:
            image_pil, mask_pil = self.augment(image_pil, mask_pil)
        image_tensor = self.transform_image(image_pil)  # [3, H, W]
        mask_tensor = self.transform_mask(mask_pil)       # [1, H, W]
        if image_tensor.shape[0] == 3:
            zeros = torch.zeros((2, *image_tensor.shape[1:]))
            image_tensor = torch.cat([image_tensor, zeros], dim=0)
        filename = os.path.basename(image_path)
        return image_tensor, mask_tensor, filename

    def __len__(self):
        """返回数据集大小"""
        return len(self.database)
