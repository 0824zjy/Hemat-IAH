import sys 
import os
import torch as th
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import time  # 导入 time 模块用于计时

# 手动将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from guided_diffusion.lidcloader import HematDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion import logger


def main():
    args = create_argparser().parse_args()
    logger.configure()

    # 确认设备（自动选择 CUDA 或 CPU）
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 1. 加载模型
    logger.log("Loading model...")
    start_time = time.perf_counter()  # 记录模型加载开始时间
    model, diffusion, _, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 直接加载完整的 state_dict
    load_model_full_state(model, args.model_path)

    model.to(device).eval()

    end_time = time.perf_counter()  # 记录模型加载结束时间
    loading_time = end_time - start_time
    logger.log(f"Model loaded in {loading_time:.2f} seconds.")

    # 2. 过滤验证集图像路径
    val_image_paths = load_val_image_paths(args.data_split)
    
    # 修改：传入正确的参数名
    # Modify the dataset initialization line to use the correct argument names
    dataset = HematDataset(train_split_file=args.data_split, val_split_file=args.data_split, test_flag=True, filter_paths=val_image_paths)


    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    logger.log("Starting inference on validation set...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 存储结果
    dices, ious = [], []

    # 记录推理开始时间
    inference_start_time = time.perf_counter()

    for i, (image, mask, filename) in enumerate(dataloader):
        batch_start_time = time.perf_counter()  # 记录每个批次开始时间

        image, mask = image.to(device), mask.to(device)
        timesteps = th.randint(0, diffusion.num_timesteps, (image.shape[0],), device=device)

        with th.no_grad():
            output = model(image, timesteps)
            pred_mask = (th.sigmoid(output) > 0.5).float()

        # 计算 Dice 和 IOU
        dice = dice_coefficient(pred_mask, mask)
        iou = iou_coefficient(pred_mask, mask)
        dices.append(dice.item())
        ious.append(iou.item())

        # 保存结果（原图、预测框、真实框）
        save_result_with_legend(image, mask, pred_mask, filename[0], args.output_dir)

        # 记录每个批次结束时间
        batch_end_time = time.perf_counter()
        batch_time = batch_end_time - batch_start_time

        if i % 10 == 0:  # 每10步打印一次
            logger.log(f"Step {i}: Dice = {dice.item():.4f}, IOU = {iou.item():.4f}, Batch Time = {batch_time:.2f} seconds")

    # 记录推理结束时间
    inference_end_time = time.perf_counter()
    total_inference_time = inference_end_time - inference_start_time
    avg_inference_time = total_inference_time / len(dataloader)

    avg_dice = sum(dices) / len(dices)
    avg_iou = sum(ious) / len(ious)
    logger.log(f"Average Dice: {avg_dice:.4f}, Average IOU: {avg_iou:.4f}")
    logger.log(f"Total Inference Time: {total_inference_time:.2f} seconds")
    logger.log(f"Average Time per Batch: {avg_inference_time:.2f} seconds")

    logger.log("Inference completed!")


def create_argparser():
    """创建参数解析器"""
    # 首先获取模型和扩散的默认参数
    defaults = model_and_diffusion_defaults()

    # 然后覆盖或添加特定于推理的参数
    inference_defaults = dict(
        model_path="/zjy/Test/results/best_model.pt",  # 使用训练后的最佳模型路径
        data_split="/zjy/Test/val_split.txt",
        output_dir="/zjy/Test/results/inference",
        image_size=64,            # 修改为训练时使用的值
        num_channels=128,         # 修改为训练时使用的值
        num_res_blocks=2,         # 修改为训练时使用的值
        num_heads=4,              # 修改为训练时使用的值
        attention_resolutions="16,8",  # 修改为训练时使用的值
        channel_mult="1,2,3,4",   # 明确设置 channel_mult
        dropout=0.0,              # 修改为训练时使用的值
        diffusion_steps=1000,     # 关键参数：时间步数
        noise_schedule="linear",
        use_fp16=False,           # 根据训练时的设置调整
        batch_size=4               # 推理时通常保持较小的 batch_size
    )

    # 更新默认参数
    defaults.update(inference_defaults)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_model_full_state(model, model_path):
    """完全加载模型参数，确保模型配置一致"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = th.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
        logger.log("Model loaded successfully.")
    except th.nn.modules.module.BadStateDictException as e:
        logger.log(f"Error loading state_dict: {e}")
        raise e
    except RuntimeError as e:
        logger.log(f"RuntimeError loading state_dict: {e}")
        raise e


def load_val_image_paths(split_file):
    """加载验证集图像路径"""
    with open(split_file, 'r', encoding='utf-8') as f:
        val_paths = [line.strip() for line in f.readlines() if line.startswith("Data/IAH")]
    return val_paths


def dice_coefficient(pred, target, smooth=1e-6, threshold=0.5):
    """计算 DICE 系数"""
    pred = th.sigmoid(pred)  # 对输出 logits 取 sigmoid
    pred = (pred > threshold).float()  # 根据阈值二值化预测

    intersection = (pred * target).sum(dim=(2, 3))  # 交集
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # 并集

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()  # 返回批次平均 DICE


def iou_coefficient(pred, target, smooth=1e-6, threshold=0.5):
    """计算 IOU 指标"""
    pred = th.sigmoid(pred)  # 对输出 logits 取 sigmoid
    pred = (pred > threshold).float()  # 根据阈值二值化预测

    intersection = (pred * target).sum(dim=(2, 3))  # 交集
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection  # 并集

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()  # 返回批次平均 IOU


def save_result_with_legend(image, gt_mask, pred_mask, filename, output_dir):
    """在原图上叠加真实框与预测框，保存带图例的结果，掩码为黑白色"""
    # 转换图像和掩码为 NumPy 数组
    image_np = image[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]  # 取第一个通道
    gt_mask_np = gt_mask[0].cpu().numpy()[0]  # 真实掩码
    pred_mask_np = pred_mask[0].cpu().numpy()[0]  # 预测掩码

    # 转换为 PIL 图像
    original_img = Image.fromarray((image_np * 255).astype(np.uint8)).convert("L")
    true_mask = Image.fromarray((gt_mask_np * 255).astype(np.uint8)).convert("L")
    pred_mask = Image.fromarray((pred_mask_np * 255).astype(np.uint8)).convert("L")

    # 在原图上标注预测掩码区域
    overlay = original_img.convert("RGBA")
    overlay_data = overlay.load()
    pred_mask_data = pred_mask.load()
    for y in range(pred_mask.height):
        for x in range(pred_mask.width):
            if pred_mask_data[x, y] > 128:
                overlay_data[x, y] = (255, 0, 0, 150)  # 半透明红色

    # 创建组合图像
    combined_width = original_img.width * 4
    combined_height = original_img.height + 30  # 添加标题高度
    combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    
    # 粘贴各部分
    combined.paste(original_img.convert("RGB"), (0, 30))
    combined.paste(true_mask.convert("RGB"), (original_img.width, 30))
    combined.paste(pred_mask.convert("RGB"), (original_img.width * 2, 30))
    combined.paste(overlay.convert("RGB"), (original_img.width * 3, 30))

    # 添加标题
    draw = ImageDraw.Draw(combined)
    title_text = "Original | True Mask | Predicted Mask | Overlay"
    try:
        # 尝试加载默认字体
        font = ImageFont.load_default()
    except:
        font = None
    draw.text((10, 5), title_text, fill=(0, 0, 0), font=font)

    # 使用图像名称保存，附加 'test'
    save_name = f"{os.path.splitext(filename)[0]}_test.png"
    combined.save(os.path.join(output_dir, save_name))
    logger.log(f"Saved result with legend to {os.path.join(output_dir, save_name)}")


if __name__ == "__main__":
    main()
