#!/usr/bin/env python3
import argparse
import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast 

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau

# ★ 모델 클래스 import (같은 폴더에 mobilenetv3s_parking_model.py 존재 필수)
from mobilenetv3s_parking_model import MultiCamParkingModel

# A100 TensorCore 활용 설정
torch.set_float32_matmul_precision('high')

# -------------------------
# 설정
# -------------------------
IMG_WIDTH = 224
IMG_HEIGHT = 224

# -------------------------
# Dataset 정의 (사용자 CSV 형식 맞춤)
# -------------------------
class MultiCamDrivingDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # CSV 구조: path, front_cam, rear_cam, left_cam, right_cam, linear_x, angular_z
        # 경로 생성: root_dir + path(폴더) + cam(파일명)
        base_path = str(row['path']).strip()
        
        # 순서 중요: Front, Left, Right, Rear (모델과 약속된 순서)
        cam_cols = ['front_cam', 'rear_cam', 'left_cam', 'right_cam']
        
        images = []
        for col in cam_cols:
            file_name = str(row[col]).strip()
            
            # 전체 경로 결합
            # 예: /data/root + augment_dataset/run_000/... + front_cam/000000.jpg
            full_path = os.path.join(self.root_dir, base_path, file_name)
            
            img = cv2.imread(full_path)
            if img is None:
                # 이미지가 깨지거나 없을 경우 검정색으로 대체 (학습 중단 방지)
                # 실제 학습시에는 로그를 찍어 확인해보는 것이 좋음
                img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            # (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        
        # Stack: (4, 3, H, W)
        images_np = np.stack(images, axis=0)
        images_tensor = torch.tensor(images_np, dtype=torch.float32)
        
        # Label
        label_vals = row[['linear_x', 'angular_z']].values.astype(np.float32)
        label_tensor = torch.tensor(label_vals, dtype=torch.float32)
        
        return images_tensor, label_tensor

# -------------------------
# Utils: Learning Rate Calculator (Tensorboard용)
# -------------------------
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# -------------------------
# Train / Eval Functions
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, scheduler=None, is_onecycle=False):
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True) # A100 데이터 전송 최적화
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision (A100 성능 극대화)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if is_onecycle and scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
    return running_loss / len(loader.dataset)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True, help="CSV 기준 최상위 데이터 폴더 경로 ex) aug_valet_parking")
    parser.add_argument('--epochs', type=int, default=100)
    # A100 권장 배치 사이즈: 128 ~ 256 (메모리에 따라 조절)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='onecycle') # onecycle 권장
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default='home/elicer/hyun/E2E/parking/trained/model')
    
    args = parser.parse_args()
    args.out_dir = os.path.expanduser(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} (A100 Optimization Enabled)")

    # CSV 로드
    print(f"📂 Reading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # 데이터 분할
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Dataset & DataLoader
    # num_workers=16: A100은 연산이 빨라서 CPU가 데이터를 빨리 줘야 함
    train_ds = MultiCamDrivingDataset(train_df, args.img_root)
    val_ds   = MultiCamDrivingDataset(val_df, args.img_root)
    test_ds  = MultiCamDrivingDataset(test_df, args.img_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=16, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=16, pin_memory=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=16, pin_memory=True)

    # 모델 생성
    model = MultiCamParkingModel(pretrained=False).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() 

    # Scheduler 설정
    scheduler = None
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                               steps_per_epoch=len(train_loader), epochs=args.epochs)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # TensorBoard
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

    best_val_loss = float('inf')
    early_stop_cnt = 0

    print(f"\n🔥 Start Training on {device}...")
    for epoch in range(args.epochs):
        is_onecycle = (args.scheduler == 'onecycle')
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, scheduler, is_onecycle)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        
        if args.scheduler == 'cosine':
            scheduler.step()
        elif args.scheduler == 'plateau':
            scheduler.step(val_loss)

        # Logging
        current_lr = get_lr(optimizer)
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {current_lr:.8f}")

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            early_stop_cnt = 0
            print("✅ Best Model Saved!")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"🛑 Early Stopping at epoch {epoch+1}")
                break
    
    writer.close()
    print("🏁 Done.")

if __name__ == "__main__":
    main()