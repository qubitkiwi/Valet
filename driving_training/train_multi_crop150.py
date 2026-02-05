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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')  # A100 가속 활성화


from mobilenetv3_multi import MultiTaskDrivingModel
from model_utils import calculate_new_lr, confusion_matrix_to_tensorboard_image 
# MultiTaskDataset은 아래에서 새로 정의한 클래스로 대체하므로 import에서 사용하지 않습니다.


# =========================
# [추가] Custom Dataset Class (Crop Top 150 & Resize)
# =========================
class CroppedMultiTaskDataset(Dataset):
    def __init__(self, X_paths, y_controls, y_signs, base_path, input_size=(224, 224)):
        """
        Args:
            X_paths: 이미지 파일명 리스트
            y_controls: 조향/속도 라벨 (Regression)
            y_signs: 표지판 라벨 (Classification)
            base_path: 이미지 경로 prefix
            input_size: 모델 입력 크기 (width, height) - MobileNetV3 등은 보통 224
        """
        self.X_paths = X_paths
        self.y_controls = y_controls
        self.y_signs = y_signs
        self.base_path = base_path
        self.input_size = input_size
        
        # ImageNet 정규화
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        # 1. 이미지 로드
        img_name = self.X_paths[idx]
        full_path = os.path.join(self.base_path, img_name)
        
        image = cv2.imread(full_path)
        if image is None:
            # 로드 실패 시 검은색 이미지 반환 (에러 방지)
            image = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # [핵심 수정] 상단 150 픽셀 자르기 (Crop Top 150)
        # ---------------------------------------------------------
        image = image[150:, :]  
        # ---------------------------------------------------------

        # 2. Resize
        # 자른 후 이미지 크기가 변하므로 모델 입력 사이즈(224x224)로 맞춤
        image = cv2.resize(image, self.input_size)

        # 3. Transform
        image = self.transform(image)

        # 4. Label 처리
        controls = torch.tensor(self.y_controls[idx], dtype=torch.float32)
        sign = torch.tensor(self.y_signs[idx], dtype=torch.long)

        return image, controls, sign


# =========================
# Settings
# =========================
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 16

FILE_NAME = '/home/elicer/song/total_data_final_v2/total_data_final_v2.csv'
DATASET_PATH = '/home/elicer/song/total_data_final_v2/'
RESULT_PATH = f'/home/elicer/song/total_data_final_v2_save_2/total_data_final_v2_reg_cls'

# [수정 1] 체크포인트 저장용 폴더 생성
CHECKPOINT_DIR = os.path.join(RESULT_PATH, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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

X_paths    = df['front_img'].values
y_controls = df[['linear_x', 'angular_z']].values
y_signs    = df['sign_class'].values.astype(np.int64)

# (1) Train+Val (90%) / Test (10%)
X_train_val, X_test, y_ctrl_train_val, y_ctrl_test, y_sign_train_val, y_sign_test = train_test_split(
    X_paths, y_controls, y_signs, test_size=0.1, random_state=42, stratify=y_signs
)

# (2) Train (72%) / Val (18%)
X_train, X_val, y_ctrl_train, y_ctrl_val, y_sign_train, y_sign_val = train_test_split(
    X_train_val, y_ctrl_train_val, y_sign_train_val, test_size=0.2, random_state=42,
    stratify=y_sign_train_val
)

# Stats 계산
ctrl_mean = y_ctrl_train.mean(axis=0).astype(np.float32)
ctrl_std  = y_ctrl_train.std(axis=0).astype(np.float32)
ctrl_std  = np.clip(ctrl_std, 1e-6, None)
print("Control mean:", ctrl_mean)
print("Control std :", ctrl_std)

# [수정 2] MultiTaskDataset -> CroppedMultiTaskDataset 교체
# 입력 사이즈 224x224 지정
train_dataset = CroppedMultiTaskDataset(X_train, y_ctrl_train, y_sign_train, DATASET_PATH, input_size=(224, 224))
val_dataset   = CroppedMultiTaskDataset(X_val,   y_ctrl_val,   y_sign_val,   DATASET_PATH, input_size=(224, 224))
test_dataset  = CroppedMultiTaskDataset(X_test,  y_ctrl_test,  y_sign_test,  DATASET_PATH, input_size=(224, 224))

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
num_sign = int(len(np.unique(y_signs)))

model = MultiTaskDrivingModel(num_signs=num_sign).to(device)

criterion_reg = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()

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
# 3) Train Loop (Classic)
# =========================
train_losses, val_losses = [], []
best_val_loss = float('inf')
early_stop_counter = 0

print("🚀 학습 시작 (Cropped Top 150)...")

for epoch in range(EPOCHS):
    # -------- Train --------
    model.train()
    running_total = 0.0
    running_reg   = 0.0
    running_cls   = 0.0

    for images, controls, signs in train_loader:
        images   = images.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)  # [B,2]
        signs    = signs.to(device, non_blocking=True)     # [B]

        optimizer.zero_grad()

        outputs = model(images)
        pred_ctrl = outputs["control"]
        pred_sign = outputs["signs"]

        loss_reg = criterion_reg(pred_ctrl, controls)
        loss_cls = criterion_cls(pred_sign, signs)

        # Classic total loss
        total_loss = loss_reg + loss_cls

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        running_total += total_loss.item()
        running_reg   += loss_reg.item()
        running_cls   += loss_cls.item()

    avg_train_total = running_total / len(train_loader)
    avg_train_reg   = running_reg   / len(train_loader)
    avg_train_cls   = running_cls   / len(train_loader)

    # -------- Val --------
    model.eval()
    val_total = 0.0
    val_reg   = 0.0
    val_cls   = 0.0
    correct, n = 0, 0

    all_preds, all_tgts = [], []
    val_lin_mse = 0.0
    val_ang_mse = 0.0

    with torch.no_grad():
        for images, controls, signs in val_loader:
            images   = images.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)
            signs    = signs.to(device, non_blocking=True)

            outputs = model(images)
            pred_ctrl = outputs["control"]
            pred_sign = outputs["signs"]

            loss_reg = criterion_reg(pred_ctrl, controls)
            loss_cls = criterion_cls(pred_sign, signs)
            total_loss = loss_reg + loss_cls

            val_total += total_loss.item()
            val_reg   += loss_reg.item()
            val_cls   += loss_cls.item()

            pred_label = pred_sign.argmax(dim=1)
            correct += (pred_label == signs).sum().item()
            n += signs.numel()

            all_preds.append(pred_label.detach().cpu())
            all_tgts.append(signs.detach().cpu())

            lin_err = (pred_ctrl[:, 0] - controls[:, 0]) ** 2
            ang_err = (pred_ctrl[:, 1] - controls[:, 1]) ** 2
            val_lin_mse += lin_err.mean().item()
            val_ang_mse += ang_err.mean().item()

    avg_val_total = val_total / len(val_loader)
    avg_val_reg   = val_reg   / len(val_loader)
    avg_val_cls   = val_cls   / len(val_loader)
    val_acc = correct / max(n, 1)

    avg_val_lin_mse = val_lin_mse / len(val_loader)
    avg_val_ang_mse = val_ang_mse / len(val_loader)

    # -------- Metrics --------
    all_preds = torch.cat(all_preds, dim=0)
    all_tgts  = torch.cat(all_tgts, dim=0)

    C = int(num_sign)
    cm = torch.zeros(C, C, dtype=torch.int64)
    for t, p in zip(all_tgts.tolist(), all_preds.tolist()):
        cm[t, p] += 1

    per_class_acc = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()).cpu().numpy()

    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp(min=1.0)
    rec  = tp / (tp + fn).clamp(min=1.0)
    f1   = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    macro_f1 = f1.mean().item()

    train_losses.append(avg_train_total)
    val_losses.append(avg_val_total)

    lr_now = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], LR:{lr_now:.6f} | \n"
        f"Train_Total:{avg_train_total:.4f} (Reg:{avg_train_reg:.4f} | Cls:{avg_train_cls:.4f}) \n"
        f"Val_Total:{avg_val_total:.4f} (Reg:{avg_val_reg:.4f} | Cls:{avg_val_cls:.4f}) \n"
        f"Val_Acc:{val_acc:.3f} | MacroF1:{macro_f1:.3f} | per-class-acc:{np.round(per_class_acc,3)}"
    )

    # -------- TensorBoard Logging --------
    writer.add_scalars('Loss/Total', {'Train': avg_train_total, 'Val': avg_val_total}, epoch)
    writer.add_scalars('Loss/Regression', {'Train': avg_train_reg, 'Val': avg_val_reg}, epoch)
    writer.add_scalars('Loss/Classification', {'Train': avg_train_cls, 'Val': avg_val_cls}, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    writer.add_scalar("Val/MacroF1", macro_f1, epoch)
    writer.add_scalar('Learning_Rate', lr_now, epoch)
    writer.add_scalar("Val/Reg_MSE_linear_x", avg_val_lin_mse, epoch)
    writer.add_scalar("Val/Reg_MSE_angular_z", avg_val_ang_mse, epoch)

    CM_LOG_EVERY = 5
    if (epoch % CM_LOG_EVERY) == 0:
        cm_img = confusion_matrix_to_tensorboard_image(cm)
        writer.add_image("Val/ConfusionMatrix", cm_img, epoch)

    # -------- [수정 2] Save Every Checkpoint --------
    # 매 에폭마다 저장합니다 (Resume을 위해 optimizer, scheduler 포함 권장)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1:03d}.pth')
    torch.save(
        {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "num_signs": num_sign,
            "val_loss": avg_val_total
        },
        ckpt_path
    )
    print(f"💾 Checkpoint saved: {ckpt_path}")

    # -------- Save Best & Early Stop --------
    # Best 모델은 그대로 덮어쓰기 방식으로 저장 (나중에 Test시 로드용)
    if avg_val_total < best_val_loss:
        best_val_loss = avg_val_total
        torch.save(
            {
                "model": model.state_dict(),
                "num_signs": num_sign,
                "epoch": epoch + 1
            },
            os.path.join(RESULT_PATH, 'best_model.pth')
        )
        early_stop_counter = 0
        print("✅ Best Model Saved (best_model.pth updated)")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

writer.close()


# =========================
# 4) Test
# =========================
# 테스트는 여전히 가장 성능이 좋았던 'best_model.pth'를 불러와서 수행합니다.
ckpt = torch.load(os.path.join(RESULT_PATH, 'best_model.pth'), map_location=device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

t_total, t_reg, t_cls = 0.0, 0.0, 0.0
correct, n = 0, 0

with torch.no_grad():
    for images, controls, signs in test_loader:
        images   = images.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)
        signs    = signs.to(device, non_blocking=True)

        outputs = model(images)
        pred_ctrl = outputs["control"]
        pred_sign = outputs["signs"]

        loss_reg = criterion_reg(pred_ctrl, controls)
        loss_cls = criterion_cls(pred_sign, signs)
        loss = loss_reg + loss_cls

        t_total += loss.item()
        t_reg   += loss_reg.item()
        t_cls   += loss_cls.item()

        pred_label = pred_sign.argmax(dim=1)
        correct += (pred_label == signs).sum().item()
        n += signs.numel()

avg_test_total = t_total / len(test_loader)
avg_test_reg   = t_reg   / len(test_loader)
avg_test_cls   = t_cls   / len(test_loader)
test_acc = correct / max(n, 1)

print(f"\n🏆 Test Total Loss:{avg_test_total:.4f} | Reg:{avg_test_reg:.4f} | Cls:{avg_test_cls:.4f} | Acc:{test_acc:.3f}")


# =========================
# 5) Learning curve plot
# =========================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('PyTorch Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plot_path = os.path.join(RESULT_PATH, 'learning_curve_pytorch.png')
plt.savefig(plot_path)
print(f"📊 학습 곡선이 '{plot_path}'에 저장되었습니다.")