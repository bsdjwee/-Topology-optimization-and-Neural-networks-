#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import TODiffusionDataset
import warnings
warnings.filterwarnings("ignore")
import random
import csv


# ==============================
# CNN 模型定义
# ==============================

class EnhancedCNN(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=1,
        base_dim=64,
        dropout=0.05,
        use_groupnorm=True
    ):
        super().__init__()

        def norm_layer(ch):
            if use_groupnorm:
                groups = min(8, ch)
                return nn.GroupNorm(groups, ch)
            else:
                return nn.BatchNorm2d(ch)

        class ResidualBlock(nn.Module):
            def __init__(self, ch, dilation=1):
                super().__init__()
                padding = dilation
                self.block = nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=padding, dilation=dilation),
                    norm_layer(ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=padding, dilation=dilation),
                    norm_layer(ch)
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(x + self.block(x))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            norm_layer(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            norm_layer(base_dim),
            nn.ReLU(inplace=True)
        )

        self.body = nn.Sequential(
            ResidualBlock(base_dim, dilation=1),
            ResidualBlock(base_dim, dilation=1),
            ResidualBlock(base_dim, dilation=2),
            ResidualBlock(base_dim, dilation=2),
            ResidualBlock(base_dim, dilation=4),
            ResidualBlock(base_dim, dilation=4),
            ResidualBlock(base_dim, dilation=2),
            ResidualBlock(base_dim, dilation=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            norm_layer(base_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_dim // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return torch.sigmoid(x)


class ConditionedCNN(nn.Module):
    def __init__(self, bc_channels=2, load_channels=2, out_channels=1, base_dim=64):
        super().__init__()
        self.cnn = EnhancedCNN(
            in_channels=bc_channels + load_channels + 1,
            out_channels=out_channels,
            base_dim=base_dim
        )

    def forward(self, bc, load, volfrac):
        x = torch.cat([bc, load], dim=1)

        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]

        vol_exp = volfrac.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)

        x = torch.cat([x, vol_exp], dim=1)

        return self.cnn(x)


class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


# ==============================
# 训练器
# ==============================

class FullTrainingWithSplit:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cuda",
        save_dir="./training_output_cnn_split",
        lr=1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        self.criterion = SimpleLoss()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=150,
            eta_min=1e-6
        )

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_ious = []
        self.val_vol_errors = []

        self.metrics_file = os.path.join(self.save_dir, "metrics.csv")

        with open(self.metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_iou",
                "val_vol_error",
                "lr"
            ])

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            density = batch["density"].to(self.device, non_blocking=True)
            bc = batch["bc"].to(self.device, non_blocking=True)
            load = batch["load"].to(self.device, non_blocking=True)
            volfrac = batch["volfrac"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(bc, load, volfrac)

            loss = self.criterion(pred, density)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_iou = 0.0
        total_vol_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                density = batch["density"].to(self.device, non_blocking=True)
                bc = batch["bc"].to(self.device, non_blocking=True)
                load = batch["load"].to(self.device, non_blocking=True)
                volfrac = batch["volfrac"].to(self.device, non_blocking=True)

                pred = self.model(bc, load, volfrac)

                total_loss += self.criterion(pred, density).item()

                pred_bin = (pred >= 0.5).float()
                density_bin = (density >= 0.5).float()

                acc = (pred_bin == density_bin).float().mean(dim=[1, 2, 3])

                intersection = (pred_bin * density_bin).sum(dim=[1, 2, 3])
                union = ((pred_bin + density_bin) > 0).float().sum(dim=[1, 2, 3])
                iou = intersection / (union + 1e-6)

                pred_vol = pred_bin.mean(dim=[1, 2, 3])
                gt_vol = density_bin.mean(dim=[1, 2, 3])
                vol_error = torch.abs(pred_vol - gt_vol) / (gt_vol + 1e-6)

                total_acc += acc.sum().item()
                total_iou += iou.sum().item()
                total_vol_error += vol_error.sum().item()
                total_samples += density.shape[0]

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_acc / total_samples
        avg_iou = total_iou / total_samples
        avg_vol_error = total_vol_error / total_samples

        return avg_loss, avg_acc, avg_iou, avg_vol_error

    def visualize_samples(self, epoch, num_samples=4, plot_acc_curve=True):
        self.model.eval()

        for loader_name, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            batch_list = list(loader)
            batch = random.choice(batch_list)

            num_show = min(num_samples, len(batch["density"]))

            fig, axes = plt.subplots(num_show, 3, figsize=(12, 4 * num_show))

            if num_show == 1:
                axes = axes.reshape(1, -1)

            total_acc = 0.0

            with torch.no_grad():
                for i in range(num_show):
                    density = batch["density"][i:i + 1].to(self.device)
                    bc = batch["bc"][i:i + 1].to(self.device)
                    load = batch["load"][i:i + 1].to(self.device)
                    volfrac = batch["volfrac"][i:i + 1].to(self.device)

                    pred = self.model(bc, load, volfrac)

                    pred_bin = (pred >= 0.5).float()
                    density_bin = (density >= 0.5).float()

                    acc = (pred_bin == density_bin).float().mean().item()
                    total_acc += acc

                    axes[i, 0].imshow(
                        np.rot90(density[0, 0].cpu().numpy(), k=-1),
                        cmap="gray_r"
                    )
                    axes[i, 0].set_title("GT")

                    axes[i, 1].imshow(
                        np.rot90(pred[0, 0].cpu().numpy(), k=-1),
                        cmap="gray_r"
                    )
                    axes[i, 1].set_title("Pred Continuous")

                    axes[i, 2].imshow(
                        np.rot90(pred_bin[0, 0].cpu().numpy(), k=-1),
                        cmap="gray_r"
                    )
                    axes[i, 2].set_title(f"Pred Binary\nAcc={acc * 100:.2f}%")

                    for ax in axes[i]:
                        ax.axis("off")

            avg_acc = total_acc / num_show

            plt.suptitle(
                f"{loader_name.capitalize()} Samples - Epoch {epoch + 1} "
                f"(Avg Binary Acc: {avg_acc * 100:.2f}%)",
                fontsize=16,
                y=1.02
            )

            plt.tight_layout()

            filename = os.path.join(
                self.save_dir,
                f"samples_epoch_{epoch + 1:03d}_{loader_name}.png"
            )

            plt.savefig(filename)
            plt.close()

            print(
                f"{loader_name.capitalize()} prediction samples saved: {filename} "
                f"| Avg Binary Acc: {avg_acc * 100:.2f}%"
            )

        if plot_acc_curve and len(self.val_accuracies) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(
                range(1, len(self.val_accuracies) + 1),
                [a * 100 for a in self.val_accuracies],
                label="Val Avg Binary Acc (%)"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Avg Binary Accuracy (%)")
            plt.title("Validation Binary Accuracy Curve")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            acc_curve_file = os.path.join(self.save_dir, "val_accuracy_curve.png")
            plt.savefig(acc_curve_file)
            plt.close()

            print(f"Validation accuracy curve saved: {acc_curve_file}")

    def train(self, max_epochs=150):
        best_iou = -1.0

        for epoch in range(max_epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_loss = self.train_epoch()
            val_loss, val_acc, val_iou, val_vol_error = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_ious.append(val_iou)
            self.val_vol_errors.append(val_vol_error)

            print(
                f"Epoch {epoch + 1}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val Acc={val_acc:.4f}, "
                f"Val IoU={val_iou:.4f}, "
                f"Val VolErr={val_vol_error:.4f}, "
                f"LR={current_lr:.8f}"
            )

            with open(self.metrics_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_acc,
                    val_iou,
                    val_vol_error,
                    current_lr
                ])

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "val_accuracies": self.val_accuracies,
                "val_ious": self.val_ious,
                "val_vol_errors": self.val_vol_errors
            }

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(
                    checkpoint,
                    os.path.join(self.save_dir, "best_cnn_model.pt")
                )
                print(f"Best model saved. Val IoU={best_iou:.4f}")

            if (epoch + 1) % 10 == 0:
                torch.save(
                    checkpoint,
                    os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt")
                )

                self.visualize_samples(epoch)

            self.scheduler.step()

        self.plot_curves()

    def plot_curves(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "val_accuracy_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.val_ious) + 1), self.val_ious, label="Val IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Validation IoU Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "iou_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(self.val_vol_errors) + 1),
            self.val_vol_errors,
            label="Val Relative Volume Error"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Relative Volume Error")
        plt.title("Validation Relative Volume Error Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "vol_error_curve.png"))
        plt.close()

        print(f"Curves saved in: {self.save_dir}")


# ==============================
# 主程序
# ==============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "/home/hym001/MLforTOP/ml/data/train_data/"
    val_dir = "/home/hym001/MLforTOP/ml/data/test_data/"

    train_dataset = TODiffusionDataset(train_dir)
    val_dataset = TODiffusionDataset(val_dir)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val/Test samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    model = ConditionedCNN(base_dim=64)

    trainer = FullTrainingWithSplit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir="./training_output_cnn_split",
        lr=1e-3
    )

    trainer.train(max_epochs=150)


if __name__ == "__main__":
    main()
