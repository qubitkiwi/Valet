#!/usr/bin/env python3
import os
import argparse
import math
import random
from pathlib import Path
import time

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# sklearn
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Check for timm
try:
    import timm
except ImportError:
    print("❌ 'timm' library is missing. Please install it: pip install timm")
    exit()

# -------------------------------------------------
# 1. Configuration & Utils
# -------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_class_weights(dataset, subset_indices, num_classes, device):
    print("⚖️ Calculating automatic class weights from training data...")
    train_labels = dataset.df.iloc[subset_indices]['sign_class'].values.astype(int)
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    
    final_weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(classes, weights):
        if cls < num_classes:
            final_weights[cls] = w
            
    weight_tensor = torch.tensor(final_weights, dtype=torch.float32).to(device)
    print(f"✅ Calculated Class Weights: {weight_tensor.cpu().numpy()}")
    return weight_tensor

# [New] Early Stopping Class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EARLY STOPPING: Count {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'✅ Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving best model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# -------------------------------------------------
# 2. Dataset
# -------------------------------------------------
class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['front_img'].strip()
        img_path = self.root_dir / img_name
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"⚠️ Warning: Image not found {img_path}")
            image = Image.new('RGB', (224, 224))

        cmd_vel = torch.tensor([row['linear_x'], row['angular_z']], dtype=torch.float32)
        sign_cls = torch.tensor(int(row['sign_class']), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, cmd_vel, sign_cls

# -------------------------------------------------
# 3. Model
# -------------------------------------------------
class RepViTMultiHead(nn.Module):
    def __init__(self, model_name='repvit_m0_9', num_classes=4):
        super(RepViTMultiHead, self).__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        except Exception:
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            num_features = features.shape[1]
            
        self.reg_head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            
        raw_reg = self.reg_head(features)
        reg_out = torch.tanh(raw_reg) * 2.0
        cls_out = self.cls_head(features)
        return reg_out, cls_out

# -------------------------------------------------
# 4. Loss
# -------------------------------------------------
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=3, class_weights=None):
        super(AutomaticWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred_cmd, true_cmd, pred_cls, true_cls):
        loss_lin = (pred_cmd[:, 0] - true_cmd[:, 0]) ** 2
        loss_lin = loss_lin.mean()
        prec_lin = torch.exp(-self.log_vars[0])
        w_loss_lin = prec_lin * loss_lin + self.log_vars[0]

        loss_ang = (pred_cmd[:, 1] - true_cmd[:, 1]) ** 2
        loss_ang = loss_ang.mean()
        prec_ang = torch.exp(-self.log_vars[1])
        w_loss_ang = prec_ang * loss_ang + self.log_vars[1]

        loss_cls = self.cls_criterion(pred_cls, true_cls)
        prec_cls = torch.exp(-self.log_vars[2])
        w_loss_cls = prec_cls * loss_cls + self.log_vars[2]

        total_loss = w_loss_lin + w_loss_ang + w_loss_cls
        return total_loss, loss_lin, loss_ang, loss_cls

# -------------------------------------------------
# 5. Train & Eval Functions
# -------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    metrics = {'loss': 0, 'lin': 0, 'ang': 0, 'cls': 0, 'acc': 0}
    total, correct = 0, 0

    for imgs, true_cmd, true_cls in loader:
        imgs, true_cmd, true_cls = imgs.to(device), true_cmd.to(device), true_cls.to(device)

        optimizer.zero_grad()
        pred_cmd, pred_cls = model(imgs)
        
        loss, l_lin, l_ang, l_cls = criterion(pred_cmd, true_cmd, pred_cls, true_cls)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total += bs
        metrics['loss'] += loss.item() * bs
        metrics['lin'] += l_lin.item() * bs
        metrics['ang'] += l_ang.item() * bs
        metrics['cls'] += l_cls.item() * bs
        
        _, predicted = torch.max(pred_cls, 1)
        correct += (predicted == true_cls).sum().item()

    return {k: v / total for k, v in metrics.items()}, correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    metrics = {'loss': 0, 'lin': 0, 'ang': 0, 'cls': 0, 'acc': 0}
    total, correct = 0, 0

    with torch.no_grad():
        for imgs, true_cmd, true_cls in loader:
            imgs, true_cmd, true_cls = imgs.to(device), true_cmd.to(device), true_cls.to(device)
            
            pred_cmd, pred_cls = model(imgs)
            loss, l_lin, l_ang, l_cls = criterion(pred_cmd, true_cmd, pred_cls, true_cls)

            bs = imgs.size(0)
            total += bs
            metrics['loss'] += loss.item() * bs
            metrics['lin'] += l_lin.item() * bs
            metrics['ang'] += l_ang.item() * bs
            metrics['cls'] += l_cls.item() * bs
            
            _, predicted = torch.max(pred_cls, 1)
            correct += (predicted == true_cls).sum().item()

    return {k: v / total for k, v in metrics.items()}, correct / total

def final_test_evaluation(model, loader, device, class_names):
    print("\n" + "="*50)
    print("🧪 Starting Final Test Evaluation...")
    print("="*50)
    model.eval()
    all_true_lin, all_pred_lin = [], []
    all_true_ang, all_pred_ang = [], []
    all_true_cls, all_pred_cls = [], []
    
    with torch.no_grad():
        for imgs, true_cmd, true_cls in loader:
            imgs = imgs.to(device)
            pred_cmd, pred_cls = model(imgs)
            
            pred_cmd_np = pred_cmd.cpu().numpy()
            true_cmd_np = true_cmd.numpy()
            
            all_pred_lin.extend(pred_cmd_np[:, 0])
            all_true_lin.extend(true_cmd_np[:, 0])
            all_pred_ang.extend(pred_cmd_np[:, 1])
            all_true_ang.extend(true_cmd_np[:, 1])
            
            _, predicted = torch.max(pred_cls, 1)
            all_pred_cls.extend(predicted.cpu().numpy())
            all_true_cls.extend(true_cls.numpy())

    mse_lin = mean_squared_error(all_true_lin, all_pred_lin)
    mse_ang = mean_squared_error(all_true_ang, all_pred_ang)
    mae_ang = mean_absolute_error(all_true_ang, all_pred_ang)
    
    print(f"\n[🚗 Driving Metrics]")
    print(f" - Linear Vel MSE : {mse_lin:.6f}")
    print(f" - Angular Vel MSE: {mse_ang:.6f}")
    print(f" - Angular Vel MAE: {mae_ang:.6f}")
    print(f"\n[🛑 Sign Classification Metrics]")
    print(classification_report(all_true_cls, all_pred_cls, target_names=class_names, digits=4))

# -------------------------------------------------
# 7. Main Execution
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/home/elicer/song/total_data_final_v2/total_data_final_v2.csv")
    parser.add_argument("--root", type=str, default="/home/elicer/song/total_data_final_v2/")
    parser.add_argument("--out_dir", type=str, default="./repvit_result_total_data_final_v2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # [수정 1] 체크포인트 저장 폴더 생성
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Tensorboard
    log_dir = save_dir / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"📋 TensorBoard logging to: {log_dir}")

    # 1. Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = DrivingDataset(args.csv, args.root, transform=transform)
    
    # Stratified Split (7:2:1)
    print("✂️  Performing Stratified Split (7:2:1)...")
    targets = full_dataset.df['sign_class'].values
    indices = np.arange(len(full_dataset))

    train_idx, rest_idx, y_train, y_rest = train_test_split(
        indices, targets, test_size=0.3, stratify=targets, random_state=42
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        rest_idx, y_rest, test_size=1/3, stratify=y_rest, random_state=42
    )
    
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)
    
    print(f"📊 Data Split: Train({len(train_ds)}), Val({len(val_ds)}), Test({len(test_ds)})")

    # Class Weights
    auto_class_weights = get_class_weights(full_dataset, train_idx, args.num_classes, device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, drop_last=True)

    # 2. Model & Loss
    model = RepViTMultiHead(model_name='repvit_m0_9', num_classes=args.num_classes).to(device)
    criterion = AutomaticWeightedLoss(num_tasks=3, class_weights=auto_class_weights).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': 0.005}
    ], lr=args.lr)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                              steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    # [New] Initialize Early Stopping (Saves best model)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_dir / "best_model.pth")

    # 3. Training Loop
    print("🚀 Training Start...")
    for epoch in range(args.epochs):
        train_res, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_res, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        weights = torch.exp(-criterion.log_vars).detach().cpu().numpy()
        
        # Tensorboard Logging
        writer.add_scalars('Loss/Total', {'Train': train_res['loss'], 'Val': val_res['loss']}, epoch)
        writer.add_scalars('Loss/Linear', {'Train': train_res['lin'], 'Val': val_res['lin']}, epoch)
        writer.add_scalars('Loss/Angular', {'Train': train_res['ang'], 'Val': val_res['ang']}, epoch)
        writer.add_scalars('Loss/Class', {'Train': train_res['cls'], 'Val': val_res['cls']}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        writer.add_scalar('Parameters/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Parameters/Weight_Linear', weights[0], epoch)
        writer.add_scalar('Parameters/Weight_Angular', weights[1], epoch)
        writer.add_scalar('Parameters/Weight_Class', weights[2], epoch)

        print(f"Ep [{epoch+1}/{args.epochs}] "
              f"T_Loss: {train_res['loss']:.4f} | V_Loss: {val_res['loss']:.4f} | "
              f"V_Acc: {val_acc*100:.1f}%")

        # [수정 2] Save Checkpoint Every Epoch
        # AutomaticWeightedLoss의 파라미터(criterion)도 저장해야 재학습 시 정확히 복구 가능
        ckpt_path = checkpoint_dir / f"model_epoch_{epoch+1:03d}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'criterion_state_dict': criterion.state_dict(),  # Loss의 학습 가능 파라미터 저장
            'val_loss': val_res['loss']
        }, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

        # [Modified] Early Stopping Call (Handles saving best model to best_model.pth)
        early_stopping(val_res['loss'], model)
        
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    writer.close()

    # 4. Final Test
    print("\n💾 Loading Best Model for Testing...")
    # Load the best model saved by EarlyStopping
    model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location=device))
    
    class_names = [f"Class {i}" for i in range(args.num_classes)]
    final_test_evaluation(model, test_loader, device, class_names)
    
    print("✅ All processes completed.")

if __name__ == "__main__":
    main()