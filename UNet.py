#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import TODiffusionDataset
import warnings
warnings.filterwarnings('ignore')
import random

# ==============================
# UNet模型定义
# ==============================
class EnhancedUNet(nn.Module):
    """增强UNet"""
    def __init__(self, in_channels=5, out_channels=1, base_dim=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.GroupNorm(4, base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GroupNorm(4, base_dim),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, 3, padding=1),
            nn.GroupNorm(4, base_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*2, base_dim*2, 3, padding=1),
            nn.GroupNorm(4, base_dim*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, 3, padding=1),
            nn.GroupNorm(4, base_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*4, base_dim*4, 3, padding=1),
            nn.GroupNorm(4, base_dim*4),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*8, 3, padding=1),
            nn.GroupNorm(4, base_dim*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*8, base_dim*8, 3, padding=1),
            nn.GroupNorm(4, base_dim*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*8, base_dim*8, 3, padding=1),
            nn.GroupNorm(4, base_dim*8),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Conv2d(base_dim*8, base_dim*4, 3, padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_dim*8, base_dim*4, 3, padding=1),
            nn.GroupNorm(4, base_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*4, base_dim*4, 3, padding=1),
            nn.GroupNorm(4, base_dim*4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Conv2d(base_dim*4, base_dim*2, 3, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*2, 3, padding=1),
            nn.GroupNorm(4, base_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*2, base_dim*2, 3, padding=1),
            nn.GroupNorm(4, base_dim*2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Conv2d(base_dim*2, base_dim, 3, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, 3, padding=1),
            nn.GroupNorm(4, base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GroupNorm(4, base_dim),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(base_dim, out_channels, 1)
    
    def forward(self, x):
        x1 = self.enc1(x)
        p1 = self.down1(x1)
        x2 = self.enc2(p1)
        p2 = self.down2(x2)
        x3 = self.enc3(p2)
        p3 = self.down3(x3)
        b = self.bottleneck(p3)
        d3 = F.interpolate(b, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.up3(d3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        d2 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.up2(d2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        d1 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.up1(d1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)

class ConditionedUNet(nn.Module):
    def __init__(self, bc_channels=2, load_channels=2, out_channels=1, base_dim=64):
        super().__init__()

        # 原来是 2 + 2 + 1 = 5
        # 现在增加 xx, yy 两个坐标通道，所以是 7
        self.unet = EnhancedUNet(
            in_channels=bc_channels + load_channels + 1 + 2,
            out_channels=out_channels,
            base_dim=base_dim
        )

    def forward(self, bc, load, volfrac):
        B, _, H, W = bc.shape
    
        vol_exp = volfrac.view(B, 1, 1, 1).expand(B, 1, H, W)
    
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=bc.device),
            torch.linspace(-1, 1, W, device=bc.device),
            indexing="ij"
        )
    
        xx = xx.view(1, 1, H, W).expand(B, 1, H, W)
        yy = yy.view(1, 1, H, W).expand(B, 1, H, W)
    
        x = torch.cat([bc, load, vol_exp, xx, yy], dim=1)
    
        return self.unet(x)



class SimpleLoss(nn.Module):
    """MSE损失"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        return self.mse(pred, target)



# ==============================
# 训练器
# ==============================
class FullTrainingWithSplit:
    def __init__(self, model, train_loader, val_loader, device="cuda", save_dir="./training_output", lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = SimpleLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=1e-6)
        self.train_losses, self.val_losses, self.val_accuracies = [], [], []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            density = batch["density"].to(self.device)
            bc = batch["bc"].to(self.device)
            load = batch["load"].to(self.device)
            volfrac = batch["volfrac"].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(bc, load, volfrac)
            loss = self.criterion(pred, density)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss/len(self.train_loader)
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                density = batch["density"].to(self.device)
                bc = batch["bc"].to(self.device)
                load = batch["load"].to(self.device)
                volfrac = batch["volfrac"].to(self.device)
                pred = self.model(bc, load, volfrac)
                pred_prob = torch.sigmoid(pred)
                
                # MSE损失
                total_loss += self.criterion(pred, density).item()
                
                # 固定阈值二值化
                pred_bin = (pred_prob >= 0.5).float()
                density_bin = (density >= 0.5).float()
                
                # 准确率
                acc = (pred_bin == density_bin).float().mean().item()
                total_acc += acc * density.shape[0]
                total_samples += density.shape[0]
        return total_loss/len(self.val_loader), total_acc/total_samples
    
    
    def visualize_samples(self, epoch, num_samples=4, plot_acc_curve=True):
        """在训练/验证集上随机抽样可视化，同时绘制平均二值化准确率曲线"""
        self.model.eval()
        for loader_name, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            # 随机选 batch
            batch_list = list(loader)
            batch = random.choice(batch_list)
            num_show = min(num_samples, len(batch["density"]))
    
            fig, axes = plt.subplots(num_show, 3, figsize=(12,4*num_show))
            if num_show == 1:
                axes = axes.reshape(1, -1)
    
            total_acc = 0
            with torch.no_grad():
                for i in range(num_show):
                    density = batch["density"][i:i+1].to(self.device)
                    bc = batch["bc"][i:i+1].to(self.device)
                    load = batch["load"][i:i+1].to(self.device)
                    volfrac = batch["volfrac"][i:i+1].to(self.device)
                    pred = self.model(bc, load, volfrac)
                    pred_prob = torch.sigmoid(pred)
    
                    # 二值化
                    pred_bin = (pred_prob >= 0.5).float()
                    density_bin = (density >= 0.5).float()
                    acc = (pred_bin == density_bin).float().mean().item()
                    total_acc += acc
    
                    # 可视化
                    axes[i,0].imshow(np.rot90(density[0,0].cpu().numpy(), k=-1), cmap="gray_r")
                    axes[i,0].set_title("GT")
                    axes[i,1].imshow(np.rot90(pred_prob[0,0].cpu().numpy(),k=-1), cmap="gray_r")
                    axes[i,1].set_title("Pred Continuous")
                    axes[i,2].imshow(np.rot90(pred_bin[0,0].cpu().numpy(), k=-1), cmap="gray_r")
                    axes[i,2].set_title(f"Pred Binary\nAcc={acc*100:.2f}%")
                    for ax in axes[i]:
                        ax.axis("off")
            avg_acc = total_acc / num_show
            plt.suptitle(f"{loader_name.capitalize()} Samples - Epoch {epoch+1} (Avg Binary Acc: {avg_acc*100:.2f}%)", fontsize=16, y=1.02)
            plt.tight_layout()
            filename = os.path.join(self.save_dir, f"samples_epoch_{epoch+1:03d}_{loader_name}.png")
            plt.savefig(filename)
            plt.close()
            print(f"{loader_name.capitalize()} prediction samples saved: {filename} | Avg Binary Acc: {avg_acc*100:.2f}%")
    
        # 绘制平均准确率曲线
        if plot_acc_curve and len(self.val_accuracies) > 0:
            plt.figure(figsize=(8,5))
            plt.plot(range(1, len(self.val_accuracies)+1), [a*100 for a in self.val_accuracies], label="Val Avg Binary Acc (%)")
            plt.xlabel("Epoch")
            plt.ylabel("Avg Binary Accuracy (%)")
            plt.title("Validation Binary Accuracy Curve")
            plt.grid(True)
            plt.legend()
            acc_curve_file = os.path.join(self.save_dir,"val_accuracy_curve.png")
            plt.tight_layout()
            plt.savefig(acc_curve_file)
            plt.close()
            print(f"Validation accuracy curve saved: {acc_curve_file}")
        
    def train(self, max_epochs=150):
        for epoch in range(max_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            self.scheduler.step() 
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # 自动保存检查点
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "val_accuracies": self.val_accuracies
            }
            torch.save(checkpoint, os.path.join(self.save_dir,f"checkpoint_epoch_{epoch+1:03d}.pt"))
            
            self.visualize_samples(epoch)
    
        # ====================================
        # 训练完成后绘制loss曲线
        # ====================================
        plt.figure(figsize=(8,5))
        plt.plot(range(1,len(self.train_losses)+1), self.train_losses, label="Train Loss")
        plt.plot(range(1,len(self.val_losses)+1), self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir,"loss_curve.png"))
        plt.close()
        print(f"Loss curve saved: {os.path.join(self.save_dir,'loss_curve.png')}")


# ==============================
# 主程序
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # train / test 数据路径
    # =========================
    train_dir = "/home/hym001/MLforTOP/ml/data/train_data/"
    test_dir  = "/home/hym001/MLforTOP/ml/data/test_data/"

    # =========================
    # 数据集
    # =========================
    train_dataset = TODiffusionDataset(train_dir)
    val_dataset = TODiffusionDataset(test_dir)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # =========================
    # DataLoader
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # =========================
    # 模型
    # =========================
    model = ConditionedUNet(base_dim=32)

    trainer = FullTrainingWithSplit(
        model,
        train_loader,
        val_loader,
        device=device,
        save_dir="./training_output_new"
    )

    trainer.train(max_epochs=150)


if __name__ == "__main__":
    main()
