#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from dataset import TODiffusionDataset
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 高效轻量 GridGCN 层
# ==============================
class FastGridGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8 if out_dim>=8 else 1, out_dim)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x

# ==============================
# 高效 GridGCN 模型
# ==============================
class FastGridGCN(nn.Module):
    """
    输入: [B, 7, H, W]  => bc+load+volfrac+xx+yy
    输出: [B, 1, H, W]
    """
    def __init__(self, in_channels=7, out_channels=1, hidden_dim=64, num_layers=6, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([
            FastGridGCNLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1),
            nn.GroupNorm(8, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        pred_residual = self.output_head(x)
        return torch.sigmoid(pred_residual)

# ==============================
# Conditioned GCN
# ==============================
class ConditionedGCN(nn.Module):
    """
    forward(bc, load, volfrac)
    """
    def __init__(self, bc_channels=2, load_channels=2, hidden_dim=64, num_layers=6):
        super().__init__()
        # 输入通道 = bc+load+volfrac+x+y = 2+2+1+2=7
        self.gcn = FastGridGCN(in_channels=7, out_channels=1, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, bc, load, volfrac):
        B, _, H, W = bc.shape
        vol_exp = volfrac.view(B,1,1,1).expand(B,1,H,W)
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=bc.device),
            torch.linspace(-1,1,W,device=bc.device),
            indexing="ij"
        )
        xx = xx.view(1,1,H,W).expand(B,1,H,W)
        yy = yy.view(1,1,H,W).expand(B,1,H,W)
        x = torch.cat([bc, load, vol_exp, xx, yy], dim=1)  # 7通道
        return self.gcn(x)

# ==============================
# Loss
# ==============================
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)

# ==============================
# Trainer
# ==============================
class GCNTrainer:
    def __init__(self, model, train_loader, test_loader, device="cuda", save_dir="./training_output_gcn_fast", lr=1e-3, max_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-6)
        self.criterion = MSELoss()

        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
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
        return total_loss / len(self.train_loader)

    def test_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                density = batch["density"].to(self.device)
                bc = batch["bc"].to(self.device)
                load = batch["load"].to(self.device)
                volfrac = batch["volfrac"].to(self.device)

                pred = self.model(bc, load, volfrac)
                loss = self.criterion(pred, density)
                total_loss += loss.item()

                pred_bin = (pred >= 0.5).float()
                density_bin = (density >= 0.5).float()
                acc = (pred_bin == density_bin).float().mean(dim=[1,2,3])
                total_acc += acc.sum().item()
                total_samples += density.shape[0]

        return total_loss / len(self.test_loader), total_acc / total_samples

    def visualize_samples(self, epoch, num_samples=4):
        self.model.eval()
        batch_list = list(self.test_loader)
        batch = random.choice(batch_list)
        num_show = min(num_samples, len(batch["density"]))

        fig, axes = plt.subplots(num_show, 3, figsize=(12, 4*num_show))
        if num_show == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for i in range(num_show):
                density = batch["density"][i:i+1].to(self.device)
                bc = batch["bc"][i:i+1].to(self.device)
                load = batch["load"][i:i+1].to(self.device)
                volfrac = batch["volfrac"][i:i+1].to(self.device)
                pred = self.model(bc, load, volfrac)
                pred_bin = (pred >= 0.5).float()
                acc = (pred_bin == (density >= 0.5).float()).float().mean().item()

                axes[i,0].imshow(np.rot90(density[0,0].cpu().numpy(), k=-1), cmap="gray_r")
                axes[i,0].set_title("GT")
                axes[i,1].imshow(np.rot90(pred[0,0].cpu().numpy(), k=-1), cmap="gray_r")
                axes[i,1].set_title("Pred Continuous")
                axes[i,2].imshow(np.rot90(pred_bin[0,0].cpu().numpy(), k=-1), cmap="gray_r")
                axes[i,2].set_title(f"Pred Binary\nAcc={acc*100:.2f}%")

                for ax in axes[i]:
                    ax.axis("off")

        plt.suptitle(f"Test Samples - Epoch {epoch+1}", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"test_samples_epoch_{epoch+1:03d}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Sample visualization saved: {save_path}")

    def save_curves(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("GCN Training/Test Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(self.test_accuracies, label="Test Binary Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("GCN Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "test_accuracy_curve.png"))
        plt.close()

    def train(self):
        best_acc = -1
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch()
            test_loss, test_acc = self.test_epoch()
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

            # 保存 checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "test_losses": self.test_losses,
                "test_accuracies": self.test_accuracies
            }
            torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1:03d}.pt"))

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pt"))
                print(f"Best model updated: Test Acc={best_acc:.4f}")

            self.visualize_samples(epoch)

# ==============================
# Main
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "/home/hym001/MLforTOP/ml/data/train_data/"
    test_dir = "/home/hym001/MLforTOP/ml/data/test_data/"

    train_dataset = TODiffusionDataset(train_dir)
    test_dataset = TODiffusionDataset(test_dir)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    model = ConditionedGCN(hidden_dim=64, num_layers=6)

    trainer = GCNTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        save_dir="./training_output_gcn_fast",
        lr=1e-3,
        max_epochs=100
    )

    trainer.train()

if __name__=="__main__":
    main()
