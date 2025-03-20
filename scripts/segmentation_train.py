import sys
import os
import argparse
import torch as th
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from guided_diffusion.lidcloader import HematDataset
from guided_diffusion.train_util import TrainLoop, logger 
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.resample import UniformSampler


class TrainLoopWithValidation(TrainLoop):
    def __init__(self, *args, val_loader=None, num_epochs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_loader = val_loader
        self.num_epochs = num_epochs

    def run_loop(self):
        best_val_iou = 0.0  
        patience = 100
        counter = 0
        for epoch in range(self.num_epochs):
            logger.log(f"Starting epoch {epoch+1}/{self.num_epochs}")
            self.train_epoch(epoch)
            val_loss, val_dice, val_iou = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_dices.append(val_dice)
            self.val_ious.append(val_iou)
            logger.log(f"Epoch {epoch+1}: Validation Loss: {val_loss:.8f}, Dice: {val_dice:.8f}, IOU: {val_iou:.8f}")

            # 当 IOU 指标提升时保存模型
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                counter = 0
                save_path = "./results/best_model.pt"
                th.save(self.model.state_dict(), save_path)
                logger.log(f"Validation IOU improved to {best_val_iou:.8f}. Saved model to {save_path}.")
            else:
                counter += 1
                logger.log(f"No improvement in validation IOU. Patience: {counter}/{patience}")
            if counter >= patience:
                logger.log("Early stopping triggered.")
                break
        self.plot_metrics()
        logger.log("Training finished!")

def main():
    args = create_argparser().parse_args()
    logger.configure()

    os.makedirs('./results', exist_ok=True)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    logger.log("Creating model and diffusion...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)

    logger.log("Creating data loader...")
    train_dataset = HematDataset(train_split_file="./train_split.txt", val_split_file="./val_split.txt", test_flag=False)
    train_loader = th.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_dataset = HematDataset(train_split_file="./train_split.txt", val_split_file="./val_split.txt", test_flag=True)
    val_loader = th.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    logger.log("Creating schedule sampler...")
    schedule_sampler = UniformSampler(diffusion, maxt=args.diffusion_steps)

    logger.log("Starting training...")
    loop = TrainLoopWithValidation(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=None,
        dataloader=train_loader,
        val_loader=val_loader, 
        prior=prior,
        posterior=posterior,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        num_epochs=args.num_epochs,  
    )

    loop.run_loop()

def create_argparser():
    defaults = dict(
        data_dir="./data/training",
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=50,
        save_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        lr_anneal_steps=1000,
        diffusion_steps=1000,
        noise_schedule="cosine",    
        image_size=256,             
        num_channels=256,           
        num_res_blocks=3,           
        num_heads=4,
        attention_resolutions="16,8",
        dropout=0.1,                
        channel_mult="1,2,3,4",
        num_epochs=200,           
        gpu="3",                  
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
