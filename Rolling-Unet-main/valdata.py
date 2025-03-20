# valdata.py
import numpy as np

import torch
import cv2
import os
from albumentations.core.transforms_interface import DualTransform

class ToTensorV2(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        print(f"Applying ToTensorV2 to image with shape: {image.shape}")
        # 转换为 CHW 格式并转为 torch.FloatTensor
        if image.ndim == 3:
            image = image.transpose(2, 0, 1).astype('float32')  # HWC to CHW
        elif image.ndim == 2:
            # 如果图像没有通道维度（灰度图），添加一个维度
            image = image[np.newaxis, :, :].astype('float32')  # H x W to 1 x H x W
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        tensor_image = torch.from_numpy(image)
        print(f"Converted image to tensor with shape: {tensor_image.shape}")
        return tensor_image

    def apply_to_mask(self, mask, **params):
        print(f"Applying ToTensorV2 to mask with shape: {mask.shape}")
        # 转换为 [1, H, W] 的 torch.FloatTensor
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :].astype('float32')  # H x W to 1 x H x W
        elif mask.ndim == 3:
            mask = mask.transpose(2, 0, 1).astype('float32')  # HWC to CHW
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        tensor_mask = torch.from_numpy(mask)
        print(f"Converted mask to tensor with shape: {tensor_mask.shape}")
        return tensor_mask

    def get_transform_init_args_names(self):
        return ()

class IAH_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, img_dir, mask_dir, num_classes, transform=None):
        self.img_paths = img_paths
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_full_path = os.path.join(self.img_dir, img_path)
        mask_full_path = os.path.join(self.mask_dir, img_path)

        img = cv2.imread(img_full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_full_path}")
        mask = mask / 255.0  # 归一化为 0-1

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 添加调试信息
        print(f"Processing image: {img_path}")
        print(f"Image tensor shape: {img.shape}")
        print(f"Mask tensor shape: {mask.shape}")

        # 确保 img 和 mask 已经是 torch.Tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected img to be torch.Tensor, but got {type(img)}")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Expected mask to be torch.Tensor, but got {type(mask)}")
        if mask.dim() != 3 or mask.size(0) != 1:
            raise ValueError(f"Expected mask to have shape [1, H, W], but got {mask.shape}")

        meta = {'img_path': img_path}  # 添加 img_path 到 meta

        return img, mask, meta
