import torch
import numpy as np

def iou_score(pred, target, smooth=1e-6):
    """
    计算交并比 (IoU)
    :param pred: 预测掩码，torch.Tensor，二值化（0 和 1）
    :param target: 真实掩码，torch.Tensor，二值化（0 和 1）
    :param smooth: 平滑因子，防止分母为零
    :return: IoU 分数，float
    """
    # 转换为布尔类型
    pred = pred.bool()
    target = target.bool()
    
    # 计算交集和并集
    intersection = torch.logical_and(pred, target).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(pred, target).sum(dim=(1, 2, 3)).float()
    
    # 计算 IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean().item()

def dice_score(pred, target, smooth=1e-6):
    """
    计算 Dice 系数
    :param pred: 预测掩码，torch.Tensor，二值化（0 和 1）
    :param target: 真实掩码，torch.Tensor，二值化（0 和 1）
    :param smooth: 平滑因子，防止分母为零
    :return: Dice 系数，float
    """
    # 转换为布尔类型
    pred = pred.bool()
    target = target.bool()
    
    # 计算交集
    intersection = torch.logical_and(pred, target).sum(dim=(1, 2, 3)).float()
    
    # 计算 Dice 系数
    dice = (2 * intersection + smooth) / (pred.sum(dim=(1, 2, 3)).float() + target.sum(dim=(1, 2, 3)).float() + smooth)
    
    return dice.mean().item()



# def iou_score(pred, target, smooth=1e-6):
#     """
#     Calculates the Intersection over Union (IoU) metric.

#     Args:
#         pred (torch.Tensor): Predicted binary mask. Shape: [batch_size, 1, H, W]
#         target (torch.Tensor): Ground truth binary mask. Shape: [batch_size, 1, H, W]
#         smooth (float): Smoothing factor to avoid division by zero.

#     Returns:
#         float: IoU score.
#     """
#     intersection = (pred * target).sum(dim=(2, 3))
#     union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     return iou.mean().item()


# def dice_score(pred, target, smooth=1e-6):
#     """
#     Calculates the Dice Coefficient metric.

#     Args:
#         pred (torch.Tensor): Predicted binary mask. Shape: [batch_size, 1, H, W]
#         target (torch.Tensor): Ground truth binary mask. Shape: [batch_size, 1, H, W]
#         smooth (float): Smoothing factor to avoid division by zero.

#     Returns:
#         float: Dice score.
#     """
#     intersection = (pred * target).sum(dim=(2, 3))
#     dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
#     return dice.mean().item()


def indicators(output, target, smooth=1e-6):
    """
    计算多个指标：IoU, Dice, HD, HD95, Recall, Specificity, Precision

    :param output: 模型输出 logits, shape [B, C, H, W]
    :param target: 真实标签, shape [B, C, H, W]
    :param smooth: 平滑项
    :return: 各指标的平均值
    """
    preds = torch.sigmoid(output)
    preds = (preds > 0.5).float()

    # IoU
    intersection = (preds * target).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou = iou.mean().item()

    # Dice
    dice = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    dice = dice.mean().item()

    # 其他指标（HD, HD95, Recall, Specificity, Precision）需要根据您的具体需求和定义进行实现
    # 这里只提供一个简单的示例，实际应用中请使用更精确的方法

    # 示例：Recall 和 Precision
    TP = (preds * target).sum(dim=(2, 3))
    FN = (target - preds * target).sum(dim=(2, 3))
    FP = (preds - preds * target).sum(dim=(2, 3))
    
    recall = (TP + smooth) / (TP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    
    recall = recall.mean().item()
    precision = precision.mean().item()

    # 示例：Specificity
    TN = ((1 - preds) * (1 - target)).sum(dim=(2, 3))
    specificity = (TN + smooth) / (TN + (1 - preds).sum(dim=(2, 3)) + smooth)
    specificity = specificity.mean().item()

    # 示例：Hausdorff Distance (HD) 和 HD95
    # 这里需要使用更专业的库如 `medpy` 或 `scikit-image`
    # 这里只是一个占位符
    hd = 0.0
    hd95 = 0.0

    return iou, dice, hd, hd95, recall, specificity, precision
