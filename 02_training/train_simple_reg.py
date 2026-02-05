#!/usr/bin/env python3
import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')  # A100 가속 활성화

# ✅ regression-only model
from mobilenetv3_reg import RegDrivingModel

from model_utils import calculate_new_lr, MultiTaskDataset  # dataset은 그대로 쓰되 sign은 무시


# =========================
# Settings
# =========================
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 16

# FILE_NAME = 'total_driving_data_20260129_merge_front_only.csv'

# DATASET_PATH = '/home/elicer/jun_ws/model_train/driving_data_20260129_merge_front_only'
# RESULT_PATH = f'/home/elicer/jun_ws/aug_model/new_data_test/new_reg'

FILE_NAME = 'driving_data_29_31.csv'

DATASET_PATH = '/home/elicer/jun_ws/model_train/driving_data_29_31'
RESULT_PATH = f'/home/elicer/jun_ws/aug_model/newer_data_test/driving_29_31_reg'


# 기준값 (LR 스케일링용)
OLD_BATCH = 128
OLD_LR = 0.001

# LR 설정
lr_linear = calculate_new_lr(OLD_LR, OLD_BATCH, BATCH_SIZE, 'linear')
lr_sqrt   = calculate_new_lr(OLD_LR, OLD_BATCH, BATCH_SIZE, 'sqrt')
target_lr = lr_sqrt

print(f"배치 사이즈: {BATCH_SIZE} 적용 LR: {target_lr:.6f}")

os.makedirs(RESULT_PATH, exist_ok=True)


# =========================
# 1) Data
# =========================
df = pd.read_csv(os.path.join(DATASET_PATH, FILE_NAME))

X_paths    = df['image_path'].values
y_controls = df[['linear_x', 'angular_z']].values.astype(np.float32)

# (1) Train+Val (90%) / Test (10%)
X_train_val, X_test, y_ctrl_train_val, y_ctrl_test = train_test_split(
    X_paths, y_controls, test_size=0.1, random_state=42
)

# (2) Train (72%) / Val (18%)
X_train, X_val, y_ctrl_train, y_ctrl_val = train_test_split(
    X_train_val, y_ctrl_train_val, test_size=0.2, random_state=42
)

# (선택) regression target stats
ctrl_mean = y_ctrl_train.mean(axis=0).astype(np.float32)
ctrl_std  = y_ctrl_train.std(axis=0).astype(np.float32)
ctrl_std  = np.clip(ctrl_std, 1e-6, None)
print("Control mean:", ctrl_mean)
print("Control std :", ctrl_std)

# ✅ MultiTaskDataset 그대로 사용 (sign label은 더미로 넣거나, dataset이 sign을 요구하면 임시로 0으로 넣기)
# - 지금 MultiTaskDataset(X, y_ctrl, y_sign, root) 형태였으니 y_sign을 dummy로 만들어 전달
dummy_train_sign = np.zeros((len(X_train),), dtype=np.int64)
dummy_val_sign   = np.zeros((len(X_val),), dtype=np.int64)
dummy_test_sign  = np.zeros((len(X_test),), dtype=np.int64)

train_dataset = MultiTaskDataset(X_train, y_ctrl_train, dummy_train_sign, DATASET_PATH)
val_dataset   = MultiTaskDataset(X_val,   y_ctrl_val,   dummy_val_sign,   DATASET_PATH)
test_dataset  = MultiTaskDataset(X_test,  y_ctrl_test,  dummy_test_sign,  DATASET_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# =========================
# 2) Model / Loss / Optim
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RegDrivingModel().to(device)

criterion_reg = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=target_lr)

scheduler = OneCycleLR(
    optimizer,
    max_lr=target_lr,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    anneal_strategy='cos',
    final_div_factor=1e4
)

# TensorBoard
log_dir = os.path.join(RESULT_PATH, 'logs')
writer = SummaryWriter(log_dir=log_dir)
print(f"📊 TensorBoard 로그 저장 경로: {log_dir}")


# =========================
# 3) Train Loop (Regression Only)
# =========================
train_losses, val_losses = [], []
best_val_loss = float('inf')
early_stop_counter = 0

print("🚀 학습 시작...")

for epoch in range(EPOCHS):
    # -------- Train --------
    model.train()
    running_reg = 0.0

    for batch in train_loader:
        # MultiTaskDataset이 (images, controls, signs)로 주는 경우 대비
        if len(batch) == 3:
            images, controls, _ = batch
        else:
            images, controls = batch

        images   = images.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)  # [B,2]

        optimizer.zero_grad()

        pred_ctrl = model(images)  # [B,2]
        loss_reg = criterion_reg(pred_ctrl, controls)

        loss_reg.backward()
        optimizer.step()
        scheduler.step()

        running_reg += loss_reg.item()

    avg_train_reg = running_reg / len(train_loader)

    # -------- Val --------
    model.eval()
    val_reg = 0.0
    val_lin_mse = 0.0
    val_ang_mse = 0.0

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, controls, _ = batch
            else:
                images, controls = batch

            images   = images.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)

            pred_ctrl = model(images)
            loss_reg = criterion_reg(pred_ctrl, controls)

            val_reg += loss_reg.item()

            lin_err = (pred_ctrl[:, 0] - controls[:, 0]) ** 2
            ang_err = (pred_ctrl[:, 1] - controls[:, 1]) ** 2
            val_lin_mse += lin_err.mean().item()
            val_ang_mse += ang_err.mean().item()

    avg_val_reg = val_reg / len(val_loader)
    avg_val_lin_mse = val_lin_mse / len(val_loader)
    avg_val_ang_mse = val_ang_mse / len(val_loader)

    train_losses.append(avg_train_reg)
    val_losses.append(avg_val_reg)

    lr_now = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], LR:{lr_now:.6f} | "
        f"Train_Reg:{avg_train_reg:.6f} | "
        f"Val_Reg:{avg_val_reg:.6f} | "
        f"Val_MSE_lin:{avg_val_lin_mse:.6f} | "
        f"Val_MSE_ang:{avg_val_ang_mse:.6f}"
    )

    # -------- TensorBoard Logging --------
    writer.add_scalars('Loss/Regression', {'Train': avg_train_reg, 'Val': avg_val_reg}, epoch)
    writer.add_scalar('Learning_Rate', lr_now, epoch)
    writer.add_scalar("Val/Reg_MSE_linear_x", avg_val_lin_mse, epoch)
    writer.add_scalar("Val/Reg_MSE_angular_z", avg_val_ang_mse, epoch)

    # -------- Save best / Early stop --------
    if avg_val_reg < best_val_loss:
        best_val_loss = avg_val_reg
        torch.save(
            {
                "model": model.state_dict(),
            },
            os.path.join(RESULT_PATH, 'best_model.pth')
        )
        early_stop_counter = 0
        print("✅ Best Model Saved")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

writer.close()


# =========================
# 4) Test
# =========================
ckpt = torch.load(os.path.join(RESULT_PATH, 'best_model.pth'), map_location=device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

t_reg = 0.0
t_lin_mse = 0.0
t_ang_mse = 0.0

with torch.no_grad():
    for batch in test_loader:
        if len(batch) == 3:
            images, controls, _ = batch
        else:
            images, controls = batch

        images   = images.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)

        pred_ctrl = model(images)
        loss_reg = criterion_reg(pred_ctrl, controls)
        t_reg += loss_reg.item()

        lin_err = (pred_ctrl[:, 0] - controls[:, 0]) ** 2
        ang_err = (pred_ctrl[:, 1] - controls[:, 1]) ** 2
        t_lin_mse += lin_err.mean().item()
        t_ang_mse += ang_err.mean().item()

avg_test_reg = t_reg / len(test_loader)
avg_test_lin_mse = t_lin_mse / len(test_loader)
avg_test_ang_mse = t_ang_mse / len(test_loader)

print(
    f"\n🏆 Test Reg Loss:{avg_test_reg:.6f} | "
    f"MSE_lin:{avg_test_lin_mse:.6f} | "
    f"MSE_ang:{avg_test_ang_mse:.6f}"
)


# =========================
# 5) Learning curve plot
# =========================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Reg Loss')
plt.plot(val_losses, label='Validation Reg Loss')
plt.title('PyTorch Regression-Only Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plot_path = os.path.join(RESULT_PATH, 'learning_curve_pytorch.png')
plt.savefig(plot_path)
print(f"📊 학습 곡선이 '{plot_path}'에 저장되었습니다.")
