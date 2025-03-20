# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']



class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        """
        初始化 BCE + Dice 损失函数。
        
        :param bce_weight: BCE 损失的权重
        :param dice_weight: Dice 损失的权重
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        """
        计算 BCE + Dice 损失。
        
        :param input: 模型的输出 logits，形状为 [B, 1, H, W]
        :param target: 真实标签，形状为 [B, 1, H, W]
        :return: 组合损失
        """
        # 计算 BCE 损失
        bce_loss = self.bce(input, target)
        
        # 计算 Dice 损失
        smooth = 1e-6
        preds = torch.sigmoid(input)
        preds = (preds > 0.5).float()
        intersection = (preds * target).sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        dice_loss = dice_loss.mean()
        
        # 组合损失
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss



class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
