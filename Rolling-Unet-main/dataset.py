# dataset.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class IAH_Dataset(Dataset):
    def __init__(self, img_paths, img_dir, mask_dir, num_classes, transform=None):
        self.img_paths = img_paths
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_relative_path = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_relative_path)
        # print(f"正在读取图像: {img_path}")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            # 使用PIL读取图像
            with Image.open(img_path) as img:
                image = img.convert('RGB')  # 确保为RGB格式
            image = np.array(image)
            # print(f"图像读取成功: {img_path}")
        except Exception as e:
            raise ValueError(f"Failed to read image: {img_path}. 错误信息: {e}")

        # 提取文件名并构建掩码文件名
        mask_filename = self._get_mask_filename(img_relative_path)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        # print(f"构建掩码路径: {mask_path}")

        if not os.path.exists(mask_path):
            # 如果掩码不存在且是负样本，创建一个全零掩码
            if 'None' in img_relative_path:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                # print(f"创建全零掩码，因为缺少掩码文件: {mask_path}")
            else:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
        else:
            try:
                with Image.open(mask_path) as msk:
                    mask = msk.convert('L')  # 灰度格式
                mask = np.array(mask)
                # 二值化掩码
                mask = (mask > 127).astype(np.uint8)
                # print(f"掩码读取成功: {mask_path}")
            except Exception as e:
                raise ValueError(f"Failed to read mask: {mask_path}. 错误信息: {e}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # [C, H, W] torch.Tensor
            mask = augmented['mask']    # [C, H, W] torch.Tensor

        # 确保掩码具有通道维度
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)  # [H, W] -> [1, H, W]
        elif mask.ndim == 3 and mask.shape[0] == 1:
            pass  # 已经是 [1, H, W]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # 转换为float张量
        mask = torch.from_numpy(mask).float()

        return image, mask, img_relative_path  # filename as string

    def _get_mask_filename(self, img_relative_path):
        """
        根据原图像文件名生成对应的掩码文件名。
        例如：'A/2023666图A-腹腔-横切-平卧位.bmp' -> '2023666图A-腹腔-横切-平卧位_mask.bmp'
        """
        base_name = os.path.basename(img_relative_path)
        name, ext = os.path.splitext(base_name)
        mask_filename = f"{name}_mask{ext}"
        return mask_filename
