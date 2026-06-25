# Hemat-IAH

面向胸腹腔出血（IAH，Intra-abdominal Hemorrhage）图像的深度学习分割实验项目。

本仓库用于比较多种医学图像分割方法在胸腹腔出血区域分割任务上的表现，当前包含：

- 标准 U-Net
- U-Net++
- Dense U-Net
- ResUNet
- Rolling-UNet（S / M / L）
- 基于 `guided_diffusion` 代码改造的带时间步嵌入 U-Net 分割流程
- 仓库说明中提到的 CIMD 实验代码框架

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 仓库结构](#2-仓库结构)
- [3. 两条主要实验流程](#3-两条主要实验流程)
- [4. 数据集组织方式](#4-数据集组织方式)
- [5. 环境安装](#5-环境安装)
- [6. 快速开始：扩散式 U-Net 分割流程](#6-快速开始扩散式-u-net-分割流程)
- [7. 快速开始：Rolling-UNet](#7-快速开始rolling-unet)
- [8. 运行 Notebook 基线实验](#8-运行-notebook-基线实验)
- [9. 模型与实现细节](#9-模型与实现细节)
- [10. 损失函数与评价指标](#10-损失函数与评价指标)
- [11. 输出文件说明](#11-输出文件说明)
- [12. 参数说明](#12-参数说明)
- [13. 复现实验建议](#13-复现实验建议)
- [14. 当前代码中的已知问题](#14-当前代码中的已知问题)
- [15. 推荐的工程化改造](#15-推荐的工程化改造)
- [16. 常见问题](#16-常见问题)
- [17. 数据安全与医学使用声明](#17-数据安全与医学使用声明)
- [18. 引用与许可证](#18-引用与许可证)

---

## 1. 项目概览

Hemat-IAH 的目标是对胸腹腔出血图像中的出血区域进行二值语义分割，并对不同网络结构进行横向比较。

输入通常是 RGB 或灰度医学图像，输出为与输入图像空间对应的二值掩码：

- `1`：出血/目标区域
- `0`：背景区域

仓库目前包含三类代码：

1. **Notebook 基线实验**

   根目录中的 `Unet.ipynb`、`UnetPlusPlus.ipynb`、`DenseUnet.ipynb` 和 `Resunet.ipynb` 是相对独立的实验 Notebook，包含数据读取、模型定义、训练、验证和可视化流程。

2. **Rolling-UNet 训练与验证工程**

   `Rolling-Unet-main/` 提供命令行训练、验证、日志记录、模型保存和预测结果可视化。

3. **扩散式/时间步条件 U-Net 分割工程**

   `guided_diffusion/` 和 `scripts/` 基于 guided-diffusion 风格代码组织模型和训练流程。当前实现会为模型随机生成 diffusion timestep，但训练目标仍然是直接的二值分割 BCE 损失。

---

## 2. 仓库结构

```text
Hemat-IAH/
├── Data/
│   └── dataset.txt
│
├── guided_diffusion/
│   ├── __init__.py
│   ├── dist_util.py
│   ├── distribution.py
│   ├── fp16_util.py
│   ├── gaussian_diffusion.py
│   ├── lidcloader.py
│   ├── logger.py
│   ├── losses.py
│   ├── nn.py
│   ├── resample.py
│   ├── respace.py
│   ├── script_util.py
│   ├── train_util.py
│   ├── unet.py
│   └── utils.py
│
├── scripts/
│   ├── segmentation_train.py
│   └── segmentation_sample.py
│
├── Rolling-Unet-main/
│   ├── README.md
│   ├── archs.py
│   ├── config.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── train_val_split.txt
│   ├── utils.py
│   ├── val.py
│   └── valdata.py
│
├── DenseUnet.ipynb
├── Resunet.ipynb
├── Unet.ipynb
├── UnetPlusPlus.ipynb
│
├── hemattrain_split.txt
├── hematval_split.txt
├── train_split.txt
├── val_split.txt
└── readme.txt
```

### 关键文件

| 文件 | 作用 |
|---|---|
| `scripts/segmentation_train.py` | 扩散式/时间步条件 U-Net 的训练入口 |
| `scripts/segmentation_sample.py` | 加载最佳权重，在验证集上推理并保存对比图 |
| `guided_diffusion/lidcloader.py` | `HematDataset` 数据集实现 |
| `guided_diffusion/script_util.py` | 创建 U-Net、Gaussian Diffusion、prior 和 posterior |
| `guided_diffusion/train_util.py` | 训练循环、验证、Dice/IoU、曲线绘制 |
| `Rolling-Unet-main/train.py` | Rolling-UNet 训练入口 |
| `Rolling-Unet-main/val.py` | Rolling-UNet 验证和预测可视化入口 |
| `Rolling-Unet-main/archs.py` | Rolling-UNet S/M/L 网络结构 |
| `Rolling-Unet-main/dataset.py` | Rolling-UNet 数据集 |
| `Rolling-Unet-main/losses.py` | BCE+Dice、Lovasz Hinge 损失 |
| `Rolling-Unet-main/metrics.py` | IoU、Dice 等评价指标 |
| `*.ipynb` | 各经典 U-Net 变体的独立实验 |

---

## 3. 两条主要实验流程

### 3.1 扩散式/时间步条件 U-Net

入口：

```bash
python scripts/segmentation_train.py
python scripts/segmentation_sample.py
```

实际训练过程：

```text
split 文件
   ↓
HematDataset
   ↓
RGB 图像 + 两个全零通道 → 5 通道输入
   ↓
随机采样 timestep
   ↓
UNetModel(image, timestep)
   ↓
2 通道 logits
   ↓
将单通道 mask 重复为 2 通道
   ↓
BCEWithLogitsLoss
   ↓
Dice / IoU 验证
   ↓
保存 best_model.pt
```

需要注意：虽然代码创建了 Gaussian Diffusion 对象，并使用 diffusion timestep 作为模型条件，但当前 `TrainLoop` 没有调用标准扩散模型常见的 `q_sample()` 或 `training_losses()` 流程，也没有对输入掩码执行显式加噪。更准确地说，它是一个**使用扩散时间步嵌入的分割 U-Net 实验**，而不是完整的 DDPM 分割训练。

### 3.2 Rolling-UNet

入口：

```bash
cd Rolling-Unet-main
python train.py
python val.py
```

训练过程：

```text
Train/Val split 文件
   ↓
IAH_Dataset
   ↓
Albumentations 数据增强与 ImageNet 标准化
   ↓
Rolling_Unet_S / M / L
   ↓
BCE+Dice 或其他损失
   ↓
混合精度 + 梯度累积
   ↓
IoU / Dice 验证
   ↓
根据验证集 IoU 保存最佳权重
```

Rolling-UNet 提供三个规模：

| 模型 | 编码通道 `embed_dims` | 定位 |
|---|---:|---|
| `Rolling_Unet_S` | `[16, 32, 64, 128, 256]` | 小型、显存占用较低 |
| `Rolling_Unet_M` | `[32, 64, 128, 256, 512]` | 中型 |
| `Rolling_Unet_L` | `[64, 128, 256, 512, 1024]` | 大型、默认实验模型 |

---

## 4. 数据集组织方式

### 4.1 推荐目录结构

建议将数据放在仓库根目录下：

```text
Data/
└── IAH/
    ├── A/
    │   ├── patient_001.bmp
    │   ├── patient_002.bmp
    │   └── ...
    │
    ├── None/
    │   ├── patient_101.bmp
    │   ├── patient_102.bmp
    │   └── ...
    │
    └── masks/
        ├── patient_001_mask.bmp
        ├── patient_002_mask.bmp
        └── ...
```

约定：

- `A/`：正样本，即存在目标区域的图像。
- `None/`：负样本，即不存在目标区域的图像。
- `masks/`：正样本对应的二值掩码。
- 正样本 `xxx.bmp` 的掩码命名为 `xxx_mask.bmp`。
- 对于 `None/` 中的负样本，代码会自动生成全零掩码，不要求存在真实掩码文件。

### 4.2 扩散式流程的 split 文件

`guided_diffusion/lidcloader.py` 读取的 split 文件要求每行是一个相对于仓库根目录的完整图像路径：

```text
Data/IAH/A/patient_001.bmp
Data/IAH/None/patient_101.bmp
Data/IAH/A/patient_002.bmp
```

训练脚本当前直接使用：

```text
./train_split.txt
./val_split.txt
```

而不是通过 `--data_dir` 自动扫描目录。

因此：

- `train_split.txt` 必须位于项目根目录。
- `val_split.txt` 必须位于项目根目录。
- 从项目根目录执行训练脚本。
- 文件中的路径必须能从当前工作目录正确解析。

### 4.3 Rolling-UNet 的 split 文件

`Rolling-Unet-main/train.py` 期望如下格式：

```text
Train:
A/patient_001.bmp
None/patient_101.bmp
A/patient_002.bmp

Val:
A/patient_003.bmp
None/patient_102.bmp
```

其中每一行是相对于 `--img_dir` 的路径。

例如：

```bash
--img_dir ../Data/IAH
--mask_dir ../Data/IAH/masks
```

则：

```text
A/patient_001.bmp
```

会被解析为：

```text
../Data/IAH/A/patient_001.bmp
```

掩码会被解析为：

```text
../Data/IAH/masks/patient_001_mask.bmp
```

> [!WARNING]
> Rolling-UNet 的掩码目录是“扁平目录”：代码只使用原图的文件名生成掩码名，不保留 `A/` 子目录。请将所有掩码直接放在 `masks/` 下。

### 4.4 掩码格式

推荐：

- 单通道灰度图
- 背景像素为 `0`
- 前景像素为 `255`
- 与原图一一对应
- 扩展名与原图一致，例如 `.bmp`

Rolling-UNet 数据集会使用阈值 `127` 将掩码二值化：

```python
mask = (mask > 127).astype(np.uint8)
```

---

## 5. 环境安装

### 5.1 克隆仓库

```bash
git clone https://github.com/0824zjy/Hemat-IAH.git
cd Hemat-IAH
```

### 5.2 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n hemat-iah python=3.9 -y
conda activate hemat-iah
```

Python 3.9 与仓库 Notebook 中记录的原始环境较接近。Python 3.10/3.11 也可能可用，但需要处理部分旧 API 的兼容性问题。

### 5.3 安装 PyTorch

请根据显卡驱动和 CUDA 版本，从 PyTorch 官方安装页面选择对应命令。

CPU 环境可使用：

```bash
pip install torch torchvision
```

GPU 环境不要盲目复制 CPU 安装命令，应安装与本机 CUDA 兼容的 PyTorch wheel。

### 5.4 安装其余依赖

仓库目前没有锁定依赖版本。根据现有 import，建议安装：

```bash
pip install \
  numpy \
  pillow \
  matplotlib \
  scikit-image \
  scikit-learn \
  pandas \
  pyyaml \
  tqdm \
  opencv-python \
  albumentations \
  timm \
  yacs \
  blobfile \
  visdom \
  warmup-scheduler \
  jupyterlab
```

可选依赖：

```bash
pip install ipywidgets
```

用于改善 Jupyter 中的进度条显示。


### 5.5 基础环境检查

```bash
python - <<'PY'
import torch
import torchvision
import numpy
import PIL
import cv2
import albumentations
import timm
import skimage

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
```

---

## 6. 快速开始：扩散式 U-Net 分割流程

### 6.1 准备数据

确保以下文件存在：

```text
Data/IAH/A/*.bmp
Data/IAH/None/*.bmp
Data/IAH/masks/*_mask.bmp
train_split.txt
val_split.txt
```

从仓库根目录检查路径：

```bash
python - <<'PY'
from pathlib import Path

for split_name in ["train_split.txt", "val_split.txt"]:
    split = Path(split_name)
    assert split.exists(), f"Missing: {split}"
    missing = []
    for line in split.read_text(encoding="utf-8").splitlines():
        path = line.strip()
        if path and not path.startswith(("Train:", "Val:")):
            if not Path(path).exists():
                missing.append(path)
    print(split_name, "missing files:", len(missing))
    for item in missing[:10]:
        print("  ", item)
PY
```

### 6.2 可选：启动 Visdom

```bash
python -m visdom.server
```

当前训练脚本虽然导入了 Visdom，但主训练流程并不依赖 Visdom 页面才能运行。

### 6.3 推荐训练命令

数据加载器默认将图像调整为 `256 × 256`，因此建议首先使用 `--image_size 256`，避免命令行模型尺寸与实际输入尺寸不一致：

```bash
python scripts/segmentation_train.py \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --microbatch -1 \
  --ema_rate 0.9999 \
  --log_interval 20 \
  --save_interval 10 \
  --use_fp16 False \
  --fp16_scale_growth 1e-3 \
  --lr_anneal_steps 1000 \
  --diffusion_steps 1000 \
  --noise_schedule cosine \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --num_heads 4 \
  --attention_resolutions 16,8 \
  --dropout 0.1 \
  --channel_mult 1,2,3,4 \
  --num_epochs 200 \
  --gpu 0
```

### 6.4 训练输出

默认输出到：

```text
results/
├── best_model.pt
├── loss_plot.png
├── dice_plot.png
└── iou_plot.png
```

最佳模型按照验证集 IoU 更新：

```python
if val_iou > best_val_iou:
    torch.save(model.state_dict(), "./results/best_model.pt")
```

早停耐心值在代码中固定为 `100` 个 epoch。

### 6.5 推理命令

推理时的网络结构参数必须与训练时完全一致：

```bash
python scripts/segmentation_sample.py \
  --model_path ./results/best_model.pt \
  --data_split ./val_split.txt \
  --output_dir ./results/inference \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --num_heads 4 \
  --attention_resolutions 16,8 \
  --channel_mult 1,2,3,4 \
  --dropout 0.1 \
  --diffusion_steps 1000 \
  --noise_schedule cosine \
  --use_fp16 False \
  --batch_size 1
```

每个样本会生成一张横向拼接图：

```text
Original | True Mask | Predicted Mask | Overlay
```

输出文件名形如：

```text
patient_003_test.png
```

### 6.6 输入与输出通道

当前代码固定：

```python
UNetModel(
    in_channels=5,
    out_channels=2,
)
```

数据加载器将三通道图像扩展为五通道：

```text
[R, G, B, 0, 0]
```

训练时，单通道 mask 被重复成两通道：

```python
target = target.repeat(1, 2, 1, 1)
```

因此当前任务实际上让两个输出通道学习同一个二值掩码。若目标是标准二值分割，更自然的配置是：

```text
输入：3 通道
输出：1 通道
标签：1 通道
```

对应修改位置：

- `guided_diffusion/lidcloader.py`
- `guided_diffusion/script_util.py`
- `guided_diffusion/train_util.py`
- `scripts/segmentation_sample.py`

---

## 7. 快速开始：Rolling-UNet

### 7.1 进入目录

```bash
cd Rolling-Unet-main
```

### 7.2 准备 split 文件

`train_val_split.txt`：

```text
Train:
A/patient_001.bmp
None/patient_101.bmp
A/patient_002.bmp

Val:
A/patient_003.bmp
None/patient_102.bmp
```

### 7.3 训练 Rolling-UNet-L

```bash
python train.py \
  --gpu_ids 0 \
  --name Rolling_Unet_L_IAH \
  --epochs 200 \
  --batch_size 4 \
  --num_workers 4 \
  --arch Rolling_Unet_L \
  --deep_supervision False \
  --input_channels 3 \
  --num_classes 1 \
  --input_w 512 \
  --input_h 512 \
  --loss BCEDiceLoss \
  --split_file ./train_val_split.txt \
  --img_dir ../Data/IAH \
  --mask_dir ../Data/IAH/masks \
  --optimizer AdamW \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --scheduler WarmupCosineAnnealingLR \
  --min_lr 1e-6 \
  --early_stopping 100 \
  --accumulation_steps 4
```

### 7.4 使用较小模型

中型：

```bash
python train.py \
  --gpu_ids 0 \
  --name Rolling_Unet_M_IAH \
  --arch Rolling_Unet_M \
  --split_file ./train_val_split.txt \
  --img_dir ../Data/IAH \
  --mask_dir ../Data/IAH/masks \
  --input_h 512 \
  --input_w 512 \
  --batch_size 4
```

小型：

```bash
python train.py \
  --gpu_ids 0 \
  --name Rolling_Unet_S_IAH \
  --arch Rolling_Unet_S \
  --split_file ./train_val_split.txt \
  --img_dir ../Data/IAH \
  --mask_dir ../Data/IAH/masks \
  --input_h 512 \
  --input_w 512 \
  --batch_size 8
```

### 7.5 显存不足时

依次尝试：

1. 使用 `Rolling_Unet_M` 或 `Rolling_Unet_S`。
2. 减小 `--batch_size`。
3. 减小 `--input_h` 和 `--input_w`。
4. 增大 `--accumulation_steps` 保持近似有效 batch size。
5. 将 `--num_workers` 调低到 `0` 或 `2` 排查数据加载问题。

例如：

```bash
python train.py \
  --gpu_ids 0 \
  --name Rolling_Unet_S_IAH_256 \
  --arch Rolling_Unet_S \
  --input_h 256 \
  --input_w 256 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --num_workers 2 \
  --split_file ./train_val_split.txt \
  --img_dir ../Data/IAH \
  --mask_dir ../Data/IAH/masks
```

### 7.6 验证和可视化

```bash
python val.py \
  --name Rolling_Unet_L_IAH \
  --split_file ./train_val_split.txt \
  --img_dir ../Data/IAH \
  --mask_dir ../Data/IAH/masks \
  --num_classes 1 \
  --batch_size 4 \
  --num_workers 4 \
  --gpu_ids 0
```

`val.py` 会读取：

```text
models/Rolling_Unet_L_IAH/config.yml
models/Rolling_Unet_L_IAH/model.pth
```

并输出：

- Validation Loss
- Validation IoU
- Validation Dice
- 预测可视化图像

当前可视化目录固定为：

```text
inference_results_unet/
```

### 7.7 Rolling-UNet 训练输出

```text
models/
└── Rolling_Unet_L_IAH/
    ├── config.yml
    ├── model.pth
    ├── log.csv
    └── metrics/
        └── metrics_curve.png
```

其中：

- `model.pth`：验证集 IoU 最优权重
- `config.yml`：本次实验参数
- `log.csv`：每个 epoch 的训练/验证指标
- `metrics_curve.png`：Loss、IoU、Dice 曲线

---

## 8. 运行 Notebook 基线实验

根目录包含：

```text
Unet.ipynb
UnetPlusPlus.ipynb
DenseUnet.ipynb
Resunet.ipynb
```

启动：

```bash
jupyter lab
```

或：

```bash
jupyter notebook
```

### 8.1 标准 U-Net Notebook

`Unet.ipynb` 中的默认配置包括：

```python
data_dir = "Data/IAH/"
split_file = "train_val_split.txt"
image_size = 304
batch_size = 4
```

其数据集规则与其他流程一致：

- `A/` 为正样本
- `None/` 为负样本
- 负样本自动生成全黑掩码
- 正样本掩码命名为 `*_mask.bmp`

### 8.2 split 文件位置

根目录当前没有明确列出的 `train_val_split.txt`，而 Notebook 默认引用该文件。运行 Notebook 前应当：

- 在根目录创建合并版 `train_val_split.txt`；或
- 修改 Notebook 中的 `split_file`；或
- 将 `Rolling-Unet-main/train_val_split.txt` 复制到根目录，并确认其中路径符合 Notebook 的解析方式。

示例：

```bash
cp Rolling-Unet-main/train_val_split.txt ./train_val_split.txt
```

复制后仍需检查路径是否真实存在。

### 8.3 推荐运行顺序

1. 运行 import 单元格。
2. 修改数据路径和训练参数。
3. 运行 split 读取和路径检查。
4. 先读取一个 batch，确认图像和掩码维度。
5. 实例化模型。
6. 用少量 epoch 做烟雾测试。
7. 确认 Loss 能下降、Dice/IoU 能更新后再进行完整训练。
8. 清除 Notebook 中保存的大型输出后再提交 Git。

---

## 9. 模型与实现细节

### 9.1 标准 U-Net

`Unet.ipynb` 使用典型编码器—解码器结构：

- 四级下采样
- 双卷积块
- Bottleneck
- 转置卷积上采样
- Skip Connection
- `1×1` 卷积输出单通道掩码

### 9.2 U-Net++

`UnetPlusPlus.ipynb` 用于实现 U-Net++ 风格的密集跳跃连接。该 Notebook 是独立实验脚本，运行参数以 Notebook 单元格中的实际配置为准。

### 9.3 Dense U-Net

`DenseUnet.ipynb` 用于实验带 Dense 连接的 U-Net 变体，以加强特征复用和梯度传递。

### 9.4 ResUNet

`Resunet.ipynb` 用于实验带残差块的 U-Net 变体，目标是改善深层网络训练稳定性。

### 9.5 Rolling-UNet

`Rolling-Unet-main/archs.py` 暴露三个模型：

```python
__all__ = [
    "Rolling_Unet_S",
    "Rolling_Unet_M",
    "Rolling_Unet_L",
]
```

网络前端包含双卷积和池化，中深层使用 `Feature_Incentive_Block` 等模块，内部还包含：

- 深度可分离卷积
- LayerNorm / BatchNorm
- GELU / ReLU
- DropPath
- 多尺度编码和解码

三个版本的主要区别是通道宽度。

### 9.6 扩散式 U-Net

`guided_diffusion/unet.py` 提供带 timestep embedding 的 U-Net。`script_util.py` 负责：

- 解析模型参数
- 创建 `UNetModel`
- 创建 `SpacedDiffusion`
- 创建 prior / posterior 的高斯编码器
---

## 10. 损失函数与评价指标

### 10.1 扩散式流程

训练损失：

```python
binary_cross_entropy_with_logits(output, target)
```

评价指标：

- Dice
- IoU

预测阈值：

```python
sigmoid(logits) > 0.5
```

### 10.2 Rolling-UNet

可选损失：

```text
BCEDiceLoss
LovaszHingeLoss
BCEWithLogitsLoss
```

默认：

```text
BCEDiceLoss
```

组合形式：

```text
Loss = bce_weight × BCE + dice_weight × DiceLoss
```

默认训练入口对 BCE 和 Dice 使用相同权重：

```python
BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
```

### 10.3 IoU

```text
IoU = |Prediction ∩ Target| / |Prediction ∪ Target|
```

### 10.4 Dice

```text
Dice = 2 × |Prediction ∩ Target| / (|Prediction| + |Target|)
```

### 10.5 空掩码样本

当预测和标签都为空时，代码中的平滑项会避免除零，并通常给出接近 1 的得分。分析结果时建议同时报告：

- 全数据集 Dice/IoU
- 正样本 Dice/IoU
- 负样本假阳性率
- 每病例统计，而不只按单张图像统计

---

## 11. 输出文件说明

### 11.1 扩散式流程

| 输出 | 说明 |
|---|---|
| `results/best_model.pt` | 验证 IoU 最优模型 |
| `results/loss_plot.png` | 训练/验证 Loss 曲线 |
| `results/dice_plot.png` | 训练/验证 Dice 曲线 |
| `results/iou_plot.png` | 训练/验证 IoU 曲线 |
| `results/inference/*_test.png` | 原图、真实掩码、预测掩码和叠加图 |

### 11.2 Rolling-UNet

| 输出 | 说明 |
|---|---|
| `models/<name>/model.pth` | 验证 IoU 最优模型 |
| `models/<name>/config.yml` | 实验配置 |
| `models/<name>/log.csv` | 每个 epoch 的指标日志 |
| `models/<name>/metrics/metrics_curve.png` | Loss/IoU/Dice 曲线 |
| `inference_results_unet/` | 验证预测可视化 |

---

## 12. 参数说明

### 12.1 扩散式训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--lr` | `1e-4` | 学习率 |
| `--weight_decay` | `1e-4` | AdamW 权重衰减 |
| `--batch_size` | `4` | batch size |
| `--microbatch` | `-1` | 小于等于 0 时等于 batch size；当前循环未真正拆分 microbatch |
| `--ema_rate` | `0.9999` | EMA 参数；当前自定义循环中未实际更新 EMA 权重 |
| `--log_interval` | `50` | 日志打印间隔 |
| `--save_interval` | `10` | 继承参数；最佳模型保存不依赖该间隔 |
| `--resume_checkpoint` | 空 | 恢复训练路径；当前自定义训练流程未完整实现恢复 |
| `--use_fp16` | `False` | 是否启用混合精度 |
| `--lr_anneal_steps` | `1000` | 当前循环未执行完整退火逻辑 |
| `--diffusion_steps` | `1000` | timestep 数量 |
| `--noise_schedule` | `cosine` | diffusion beta schedule |
| `--image_size` | `256`（训练入口覆盖） | 模型配置尺寸；数据集尺寸仍由 loader 默认值决定 |
| `--num_channels` | `256`（训练入口覆盖） | U-Net 基础通道数 |
| `--num_res_blocks` | `3`（训练入口覆盖） | 每层残差块数 |
| `--num_heads` | `4` | 注意力头数 |
| `--attention_resolutions` | `16,8` | 注意力层设置 |
| `--dropout` | `0.1` | Dropout |
| `--channel_mult` | `1,2,3,4` | 各层通道倍率 |
| `--num_epochs` | `200` | 最大 epoch |
| `--gpu` | `"3"` | 入口层选择的 GPU；底层代码仍存在 `cuda:0` 硬编码 |

### 12.2 Rolling-UNet 训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--gpu_ids` | `3` | CUDA 可见设备 |
| `--name` | 自动生成 | 实验名称 |
| `--epochs` | `200` | epoch |
| `--batch_size` | `8` | 单次 DataLoader batch 参数 |
| `--num_workers` | `8` | DataLoader 进程数 |
| `--arch` | `Rolling_Unet_L` | 模型结构 |
| `--deep_supervision` | `False` | 深监督开关 |
| `--input_channels` | `3` | 输入通道 |
| `--num_classes` | `1` | 输出类别/通道 |
| `--input_w` | `512` | 输入宽度 |
| `--input_h` | `512` | 输入高度 |
| `--loss` | `BCEDiceLoss` | 损失函数 |
| `--split_file` | `train_val_split.txt` | 数据划分文件 |
| `--img_dir` | 硬编码绝对路径 | 图像根目录 |
| `--mask_dir` | 硬编码绝对路径 | 掩码目录 |
| `--optimizer` | `AdamW` | 优化器 |
| `--lr` | `1e-4` | 学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--scheduler` | `WarmupCosineAnnealingLR` | 学习率调度器 |
| `--min_lr` | `1e-6` | 最低学习率 |
| `--early_stopping` | `100` | 早停耐心值 |
| `--accumulation_steps` | `4` | 梯度累积步数 |
| `--output_dir` | `output/` | 预留验证输出目录 |

支持的 scheduler：

```text
CosineAnnealingLR
ReduceLROnPlateau
MultiStepLR
ConstantLR
WarmupCosineAnnealingLR
```

---

## 13. 复现实验建议

### 13.1 固定患者级数据划分

医学图像可能存在同一患者的多张切面。应按患者划分训练集和验证集，避免同一患者图像出现在不同集合中造成数据泄漏。

### 13.2 记录完整实验配置

每次实验至少记录：

- Git commit
- Python 版本
- PyTorch 版本
- CUDA/cuDNN 版本
- GPU 型号
- 随机种子
- 数据版本
- 患者级划分列表
- 图像预处理方式
- 输入尺寸
- batch size
- 有效 batch size
- 优化器和学习率
- 最优 epoch
- 阈值
- 正样本/负样本分开统计结果

Rolling-UNet 已保存 `config.yml` 和 `log.csv`，建议扩散式流程也采用同样方式。

### 13.3 先做烟雾测试

完整训练前先运行：

```text
2～5 个 epoch
少量训练样本
少量验证样本
num_workers=0
batch_size=1
```

确认：

- 数据可读取
- 掩码与图像对应
- 输出尺寸正确
- Loss 有限且能反向传播
- 模型可以保存和重新加载
- 推理图像方向、尺寸、掩码叠加正确

### 13.4 统计类别分布

建议统计：

```text
正样本数量
负样本数量
前景像素比例
不同检查切面数量
每名患者的图像数量
```

在前景面积很小的情况下，仅使用 BCE 容易偏向背景，需要评估 Dice、Focal、Tversky 或边界损失。

### 13.5 推荐报告指标

建议至少报告：

- Dice
- IoU
- Precision
- Recall/Sensitivity
- Specificity
- 像素级 AUROC（可选）
- HD95（边界质量）
- 每张图像推理时间
- 参数量和显存占用

---

