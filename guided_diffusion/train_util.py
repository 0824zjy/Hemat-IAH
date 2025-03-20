import torch as th
import blobfile as bf
from torch.optim import AdamW
from guided_diffusion import logger
from guided_diffusion.resample import UniformSampler
from visdom import Visdom
from guided_diffusion.fp16_util import MixedPrecisionTrainer
import matplotlib.pyplot as plt
import os

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_coefficient(output, target, smooth=1.0):
    preds = th.sigmoid(output)
    preds = (preds > 0.5).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean()

def iou_coefficient(output, target, smooth=1.0):
    preds = th.sigmoid(output)
    preds = (preds > 0.5).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        prior=None,
        posterior=None,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion, maxt=1000)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.prior = prior
        self.posterior = posterior

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        # 模型及优化器初始化
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 混合精度训练支持
        if self.use_fp16:
            self.mp_trainer = MixedPrecisionTrainer(
                model=self.model, use_fp16=self.use_fp16, fp16_scale_growth=self.fp16_scale_growth
            )
            logger.log("Using mixed precision training")
        else:
            self.mp_trainer = None
            logger.log("Using full precision training")

        # 用于记录训练和验证过程中的指标
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.train_ious = []
        self.val_ious = []

    def train_epoch(self, epoch):
        """在一个 epoch 内使用训练集进行模型训练"""
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        for i, (batch, target, filename) in enumerate(self.dataloader):
            batch = batch.to(self.device)
            target = target.to(self.device)

            if th.isnan(batch).any() or th.isnan(target).any():
                logger.log(f"NaN detected in training data at epoch {epoch+1}, step {i}. Skipping this batch.")
                continue
            if th.isinf(batch).any() or th.isinf(target).any():
                logger.log(f"Inf detected in training data at epoch {epoch+1}, step {i}. Skipping this batch.")
                continue
            target = target.repeat(1, 2, 1, 1)

            timesteps = th.randint(
                low=0, high=self.diffusion.num_timesteps, size=(batch.shape[0],),
                device=self.device, dtype=th.long
            )

            self.opt.zero_grad()
            with th.cuda.amp.autocast(enabled=self.use_fp16):
                output = self.model(batch, timesteps)
                loss = th.nn.functional.binary_cross_entropy_with_logits(output, target)

            # 检查输出和 loss 中是否有 NaN 或 Inf
            if th.isnan(output).any() or th.isinf(output).any():
                logger.log(f"NaN or Inf detected in model output at epoch {epoch+1}, step {i}. Skipping this batch.")
                continue
            if th.isnan(loss).any() or th.isinf(loss).any():
                logger.log(f"NaN or Inf detected in loss at epoch {epoch+1}, step {i}. Skipping this batch.")
                continue

            # 计算 Dice 和 IOU 指标
            dice = dice_coefficient(output, target)
            iou = iou_coefficient(output, target)

            # 反向传播及参数更新
            if self.use_fp16 and self.mp_trainer is not None:
                self.mp_trainer.backward(loss)
                self.mp_trainer.step(self.opt)
            else:
                loss.backward()
                self.opt.step()

            # 梯度裁剪
            th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 检查梯度中是否存在异常值
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if th.isnan(param.grad).any():
                        logger.log(f"NaN detected in gradients of {name} at epoch {epoch+1}, step {i}.")
                    if th.isinf(param.grad).any():
                        logger.log(f"Inf detected in gradients of {name} at epoch {epoch+1}, step {i}.")

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            epoch_iou += iou.item()

            if i % self.log_interval == 0:
                logger.log(f"Epoch: {epoch+1}, Step: {i}, Loss: {loss.item():.8f}, Dice: {dice.item():.8f}, IOU: {iou.item():.8f}")

        avg_train_loss = epoch_loss / len(self.dataloader)
        avg_train_dice = epoch_dice / len(self.dataloader)
        avg_train_iou = epoch_iou / len(self.dataloader)
        self.train_losses.append(avg_train_loss)
        self.train_dices.append(avg_train_dice)
        self.train_ious.append(avg_train_iou)
        logger.log(f"Epoch {epoch+1}: Average Training Loss: {avg_train_loss:.8f}, Dice: {avg_train_dice:.8f}, IOU: {avg_train_iou:.8f}")

    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        with th.no_grad():
            for i, (batch, target, filename) in enumerate(eval_loader):
                batch = batch.to(self.device)
                target = target.to(self.device)

                if th.isnan(batch).any() or th.isnan(target).any():
                    logger.log(f"NaN detected in evaluation data at step {i}. Skipping this batch.")
                    continue
                if th.isinf(batch).any() or th.isinf(target).any():
                    logger.log(f"Inf detected in evaluation data at step {i}. Skipping this batch.")
                    continue

                target = target.repeat(1, 2, 1, 1)
                timesteps = th.randint(
                    low=0, high=self.diffusion.num_timesteps, size=(batch.shape[0],),
                    device=self.device, dtype=th.long
                )
                output = self.model(batch, timesteps)

                if th.isnan(output).any() or th.isinf(output).any():
                    logger.log(f"NaN or Inf detected in model output during evaluation at step {i}. Skipping this batch.")
                    continue

                loss = th.nn.functional.binary_cross_entropy_with_logits(output, target)

                if th.isnan(loss).any() or th.isinf(loss).any():
                    logger.log(f"NaN or Inf detected in loss during evaluation at step {i}. Skipping this batch.")
                    continue

                dice = dice_coefficient(output, target)
                iou = iou_coefficient(output, target)
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()

        avg_loss = total_loss / len(eval_loader)
        avg_dice = total_dice / len(eval_loader)
        avg_iou = total_iou / len(eval_loader)
        self.model.train()
        return avg_loss, avg_dice, avg_iou

    def run_loop(self):
        self.train_epoch(0)
        train_eval = self.evaluate(self.dataloader)
        logger.log(f"Training Evaluation - Loss: {train_eval[0]:.8f}, Dice: {train_eval[1]:.8f}, IOU: {train_eval[2]:.8f}")

    def plot_metrics(self):
        """绘制并保存训练和验证过程中 Loss、Dice 和 IOU 曲线"""
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, self.val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('./results/loss_plot.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_dices, label='Train Dice', color='blue')
        plt.plot(epochs, self.val_dices, label='Val Dice', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.title('Dice Coefficient over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('./results/dice_plot.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_ious, label='Train IOU', color='blue')
        plt.plot(epochs, self.val_ious, label='Val IOU', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('IOU')
        plt.title('IOU over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('./results/iou_plot.png')
        plt.close()

def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint():
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
