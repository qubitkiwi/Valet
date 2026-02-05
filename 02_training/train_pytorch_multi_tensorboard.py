import math
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')  # A100 가속 활성화
from mobilenetv3_multi import MultiTaskDrivingModel

from focal import FocalLoss
from model_utils import calculate_new_lr, MultiTaskDataset, confusion_matrix_to_tensorboard_image


# --- 설정 및 전처리 ---
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 16
CROP_HEIGHT = 100
FILE_NAME = 'total_data_sign_labeled.csv'  # 파일명
DATASET_PATH = '/home/elicer/jun_ws/model_train/sign_label_images'  # 상위 폴더
RESULT_PATH = f'/home/elicer/jun_ws/aug_model/mobilenet_v3s/Kendall_CE_take_1'

# 기준값 (LR 스케일링용)
OLD_BATCH = 128
OLD_LR = 0.001

# LR 설정
lr_linear = calculate_new_lr(OLD_LR, OLD_BATCH, BATCH_SIZE, 'linear')
lr_sqrt = calculate_new_lr(OLD_LR, OLD_BATCH, BATCH_SIZE, 'sqrt')
target_lr = lr_sqrt

print(f"배치 사이즈: {BATCH_SIZE} 적용 LR: {target_lr:.5f}")


if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


# 1. 데이터 준비
df = pd.read_csv(os.path.join(DATASET_PATH, FILE_NAME))

X_paths = df['image_path'].values
y_controls = df[['linear_x', 'angular_z']].values
y_signs = df['sign_class'].values

# (1) Train+Val (90%) 와 Test (10%) 분리
X_train_val, X_test, y_ctrl_train_val, y_ctrl_test, y_sign_train_val, y_sign_test = train_test_split(
    X_paths, y_controls, y_signs, test_size=0.1, random_state=42)

# (2) Train+Val 을 다시 Train (72%) 와 Val (18%)로 분리
X_train, X_val, y_ctrl_train, y_ctrl_val, y_sign_train, y_sign_val = train_test_split(
    X_train_val, y_ctrl_train_val, y_sign_train_val, test_size=0.2, random_state=42)


ctrl_mean = y_ctrl_train.mean(axis=0).astype(np.float32)
ctrl_std  = y_ctrl_train.std(axis=0).astype(np.float32)
ctrl_std = np.clip(ctrl_std, 1e-6, None)

print("Control mean:", ctrl_mean)
print("Control std :", ctrl_std)


train_dataset = MultiTaskDataset(X_train, y_ctrl_train, y_sign_train, DATASET_PATH)
val_dataset = MultiTaskDataset(X_val, y_ctrl_val, y_sign_val, DATASET_PATH)
test_dataset = MultiTaskDataset(X_test, y_ctrl_test, y_sign_test, DATASET_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# 2. 모델, 손실함수, 최적화 도구 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_sign = len(np.unique(y_signs))
model = MultiTaskDrivingModel(num_signs=num_sign).to(device)


# ----- regression: MSELoss -----
criterion_reg = nn.MSELoss()

# ----- Focal alpha (optional but recommended for imbalance) -----
counts = np.bincount(y_signs, minlength=num_sign).astype(np.float32)
freq = counts / max(counts.sum(), 1.0)
alpha = (1.0 / (freq + 1e-12))
alpha = alpha / alpha.sum() * num_sign
alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
print("Class counts:", counts)
print("Focal alpha:", alpha)

# criterion_cls = FocalLoss(gamma=1.0, alpha=alpha_t)
criterion_cls = nn.CrossEntropyLoss()


# ----- Kendall uncertainty params (learnable log variances) -----
log_var_reg = torch.nn.Parameter(torch.zeros(1, device=device))  # s_reg = log(sigma_reg^2)
log_var_cls = torch.nn.Parameter(torch.zeros(1, device=device))  # s_cls = log(sigma_cls^2)

optimizer = optim.Adam(
    list(model.parameters()) + [log_var_reg, log_var_cls],
    lr=target_lr
)


# 3. OneCycleLR 스케줄러 설정 (Warmup 포함)
scheduler = OneCycleLR(
    optimizer,
    max_lr=target_lr,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    anneal_strategy='cos',
    final_div_factor=1e4
)

# Tensorboard
log_dir = os.path.join(RESULT_PATH, 'logs')
writer = SummaryWriter(log_dir=log_dir)
train_losses, val_losses = [], []
print(f"📊 TensorBoard 로그 저장 경로: {log_dir}")

# 4. 학습 루프 (Early Stopping 및 History 기록 추가)
best_val_loss = float('inf')
early_stop_counter = 0


print("🚀 학습 시작...")
for epoch in range(EPOCHS):
    # =========================================== 학습(Training) ================================
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0
    running_cls_loss = 0.0

    for images, controls, signs in train_loader:
        images = images.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)
        signs = signs.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss_reg = criterion_reg(outputs['control'], controls)
        loss_cls = criterion_cls(outputs['signs'], signs)

        # ---------------- Kendall uncertainty-weighted total loss ----------------
        precision_reg = torch.exp(-log_var_reg)
        precision_cls = torch.exp(-log_var_cls)
        total_loss = 0.5 * (
            precision_reg * loss_reg + log_var_reg + 
            precision_cls * loss_cls + log_var_cls
        )
        # ------------------------------------------------------------------------

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Optional: clamp to avoid extreme values early
        with torch.no_grad():
            log_var_reg.clamp_(-3.0, 3.0)
            log_var_cls.clamp_(-3.0, 3.0)

        running_loss += total_loss.item()
        running_reg_loss += loss_reg.item()
        running_cls_loss += loss_cls.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_reg = running_reg_loss / len(train_loader)
    avg_train_cls = running_cls_loss / len(train_loader)

    # =========================================== 검증(Validation) ================================
    model.eval()

    val_total, val_total_reg, val_total_cls = 0.0, 0.0, 0.0
    correct, n = 0, 0

    # NEW: collect predictions & targets for better metrics
    all_preds = []
    all_tgts  = []

    # NEW: track per-dimension regression losses (linear_x vs angular_z)
    val_lin_mse = 0.0
    val_ang_mse = 0.0

    with torch.no_grad():
        precision_reg = torch.exp(-log_var_reg)
        precision_cls = torch.exp(-log_var_cls)

        for images, controls, signs in val_loader:
            images   = images.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)   # shape [B, 2] -> [linear_x, angular_z]
            signs    = signs.to(device, non_blocking=True)

            outputs = model(images)
            pred_ctrl = outputs["control"]   # expect shape [B, 2]
            pred_sign = outputs["signs"]     # logits [B, C]

            # raw task losses
            loss_reg = criterion_reg(pred_ctrl, controls)
            loss_cls = criterion_cls(pred_sign, signs)

            # Kendall total loss (training objective)
            total_loss = precision_reg * loss_reg + log_var_reg + precision_cls * loss_cls + log_var_cls

            val_total     += total_loss.item()
            val_total_reg += loss_reg.item()
            val_total_cls += loss_cls.item()

            # basic acc
            pred_label = pred_sign.argmax(dim=1)
            correct += (pred_label == signs).sum().item()
            n += signs.numel()

            # store for macro-F1/confusion matrix
            all_preds.append(pred_label.detach().cpu())
            all_tgts.append(signs.detach().cpu())

            # per-dim MSE (helps debugging when one dominates)
            # controls order in your code: [linear_x, angular_z]
            lin_err = (pred_ctrl[:, 0] - controls[:, 0]) ** 2
            ang_err = (pred_ctrl[:, 1] - controls[:, 1]) ** 2
            val_lin_mse += lin_err.mean().item()
            val_ang_mse += ang_err.mean().item()

    avg_val_loss = val_total / len(val_loader)
    avg_val_reg  = val_total_reg / len(val_loader)
    avg_val_cls  = val_total_cls / len(val_loader)
    val_acc = correct / max(n, 1)

    # NEW: finalize per-dim mse
    avg_val_lin_mse = val_lin_mse / len(val_loader)
    avg_val_ang_mse = val_ang_mse / len(val_loader)

    # =========================
    # NEW: confusion matrix + macro-F1 + per-class accuracy
    # =========================
    all_preds = torch.cat(all_preds, dim=0)  # [N]
    all_tgts  = torch.cat(all_tgts, dim=0)   # [N]

    C = int(num_sign)
    cm = torch.zeros(C, C, dtype=torch.int64)
    # cm[true, pred] += 1
    for t, p in zip(all_tgts.tolist(), all_preds.tolist()):
        cm[t, p] += 1

    # per-class accuracy: diag / row sum
    per_class_acc = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()).cpu().numpy()

    # macro-F1
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp(min=1.0)
    rec  = tp / (tp + fn).clamp(min=1.0)
    f1   = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    macro_f1 = f1.mean().item()


    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    lr_now = optimizer.param_groups[0]["lr"]

    with torch.no_grad():
        w_reg = torch.exp(-log_var_reg).item()
        w_cls = torch.exp(-log_var_cls).item()
        s_reg = log_var_reg.item()
        s_cls = log_var_cls.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], LR: {lr_now:.6f} | \n",
        f"Train_Total: {avg_train_loss:.4f} | Reg: {avg_train_reg:.4f} | Cls: {avg_train_cls:.4f} \n",
        f"Val_Total: {avg_val_loss:.4f} (Reg:{avg_val_reg:.4f} | Cls:{avg_val_cls:.4f}) \n",
        f"Val_Acc:{val_acc:.3f} | MacroF1:{macro_f1:.3f} | per-class-acc:{np.round(per_class_acc,3)} \n",
        f"Kendall_w(Reg:{w_reg:.3f}, Cls:{w_cls:.3f})"
    )

    # ==================== Logging ====================
    writer.add_scalars('Loss/Total', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)
    writer.add_scalars('Loss/Regression', {'Train': avg_train_reg, 'Val': avg_val_reg}, epoch)
    writer.add_scalars('Loss/Classification', {'Train': avg_train_cls, 'Val': avg_val_cls}, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    writer.add_scalar('Learning_Rate', lr_now, epoch)

    writer.add_scalar("Uncertainty/weight_reg", w_reg, epoch)
    writer.add_scalar("Uncertainty/weight_cls", w_cls, epoch)
    writer.add_scalar("Uncertainty/log_var_reg", s_reg, epoch)
    writer.add_scalar("Uncertainty/log_var_cls", s_cls, epoch)

    writer.add_scalar("Val/MacroF1", macro_f1, epoch)
    # regression per-dim mse
    writer.add_scalar("Val/Reg_MSE_linear_x", avg_val_lin_mse, epoch)
    writer.add_scalar("Val/Reg_MSE_angular_z", avg_val_ang_mse, epoch)

    # confusion matrix image (log every N epochs to reduce overhead)
    CM_LOG_EVERY = 5
    if (epoch % CM_LOG_EVERY) == 0:
        cm_img = confusion_matrix_to_tensorboard_image(cm)
        writer.add_image("Val/ConfusionMatrix", cm_img, epoch)

    # ========================= Save best / Early stop =======================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            {
                "model": model.state_dict(),
                "log_var_reg": log_var_reg.detach().cpu(),
                "log_var_cls": log_var_cls.detach().cpu(),
                "num_signs": num_sign,
            },
            os.path.join(RESULT_PATH, 'best_model.pth')
        )
        early_stop_counter = 0
        print("Best Model Saved")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

writer.close()


# 5. ========================= 최종 테스트 세트 평가 =============================
ckpt = torch.load(os.path.join(RESULT_PATH, 'best_model.pth'), map_location=device)
model.load_state_dict(ckpt["model"], strict=True)

# restore uncertainties
log_var_reg = torch.nn.Parameter(ckpt["log_var_reg"].to(device))
log_var_cls = torch.nn.Parameter(ckpt["log_var_cls"].to(device))

model.eval()
t_total, t_reg, t_cls = 0.0, 0.0, 0.0
correct, n = 0, 0

with torch.no_grad():
    precision_reg = torch.exp(-log_var_reg)
    precision_cls = torch.exp(-log_var_cls)

    for images, controls, signs in test_loader:
        images = images.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)
        signs = signs.to(device, non_blocking=True)

        outputs = model(images)
        pred_ctrl = outputs["control"]
        pred_sign = outputs["signs"]

        loss_reg = criterion_reg(pred_ctrl, controls)
        loss_cls = criterion_cls(pred_sign, signs)

        loss = precision_reg * loss_reg + log_var_reg + precision_cls * loss_cls + log_var_cls

        t_total += loss.item()
        t_reg += loss_reg.item()
        t_cls += loss_cls.item()

        pred_label = pred_sign.argmax(dim=1)
        correct += (pred_label == signs).sum().item()
        n += signs.numel()

avg_test_total = t_total / len(test_loader)
avg_test_reg = t_reg / len(test_loader)
avg_test_cls = t_cls / len(test_loader)
test_acc = correct / max(n, 1)

print(f"\n🏆 Test Total Loss:{avg_test_total:.4f} | Reg Loss:{avg_test_reg:.4f} | Cls Loss:{avg_test_cls:.4f} | Accuracy:{test_acc:.3f}")

# 6. 학습 곡선 시각화 및 저장
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