import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

# 분리한 모듈들 임포트
import config as cfg
from dataset import MultiCamDrivingDataset
import trainer
from mobilenetv3s_parking_model_pretrained_multi import MultiCamParkingModel # 기존 모델 파일

"""
python train.py \
--csv_path "/home/elicer/data/valet_parking_final/aug_final.csv" \
--img_root "/home/elicer/data/valet_parking_final" \
--out_dir_prefix "mobilenetv3s_pretrained_up_mid_crop_LR_cls_05_sampler_final" \
--epochs 100 \
--scheduler "onecycle"

python train.py \
--csv_path "/home/elicer/data/valet_parking_final/aug_final_with04.csv" \
--img_root "/home/elicer/data/valet_parking_final" \
--out_dir_prefix "mobilenetv3s_pretrained_up_mid_crop_LR_cls_05_sampler_final_with04" \
--epochs 100 \
--scheduler "onecycle"

python train.py \
--csv_path "/home/elicer/data/valet_parking_final/aug_final.csv" \
--img_root "/home/elicer/data/valet_parking_final" \
--out_dir_prefix "mobilenetv3s_pretrained_up_mid_crop_LR_cls_05_sampler_final" \
--epochs 200 \
--scheduler "onecycle"

python train.py \
--csv_path "/home/elicer/data/valet_parking_final/aug_final_with04.csv" \
--img_root "/home/elicer/data/valet_parking_final" \
--out_dir_prefix "mobilenetv3s_pretrained_up_mid_crop_LR_cls_05_sampler_final_with04" \
--epochs 200 \
--scheduler "onecycle"
"""

NUM_WORKERS = 4

def create_sampler(df, class_col_name):
    y_train = df[class_col_name].values.astype(int)

    print(f"   Sampler Check -> Column: {class_col_name}")
    print(f"   Unique Values: {np.unique(y_train)}")

    class_counts = np.bincount(y_train)
    print(f"   Class Counts in Train: {class_counts}")
    
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def main():
    # 1. Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='onecycle', choices=['onecycle', 'cosine', 'cosine_restart', 'plateau'])
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--out_dir_prefix', type=str, required=True)
    
    args = parser.parse_args()

    # 결과 저장 경로 설정
    OUT_BASE = "/home/elicer/hyun_ws/E2E/parking/trained/multi/model"
    lr_str = f"{args.lr:.3f}".replace('.', '')
    dir_name = f"{args.out_dir_prefix}_{args.scheduler}_batch{args.batch_size}_epoch{args.epochs}_lr{lr_str}"
    args.out_dir = os.path.join(OUT_BASE, dir_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("📁 Output dir:", args.out_dir)
    
    # 2. Setup
    # A100 TensorCore 활용 설정
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} (A100 Optimization Enabled)")

    # 3. Data Preparation
    print(f"📂 Reading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # ================= [필수 추가 코드] 빈 값 청소 =================
    # config에 설정된 클래스 컬럼명 가져오기
    target_col = cfg.CLASS_COL_NAME  # 'sign_class'
    
    # 1. 빈 값(NaN)이 있는지 확인
    nan_count = df[target_col].isnull().sum()
    if nan_count > 0:
        print(f"⚠️ Warning: Found {nan_count} rows with NaN (empty) in '{target_col}'. Dropping them!")
        # 빈 값이 있는 행을 아예 삭제해버림
        df = df.dropna(subset=[target_col])
        
    # 2. 인덱스 초기화 (중간에 빠진 행 정리)
    df = df.reset_index(drop=True)

    # 3. 이제 안전하게 int로 변환
    df[target_col] = df[target_col].astype(int)
    print(f"✅ Data Cleaned. Valid samples: {len(df)}")
    # =============================================================

    temp_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(temp_df, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Sampler 생성
    sampler = create_sampler(train_df, cfg.CLASS_COL_NAME)

    # DataLoader
    train_ds = MultiCamDrivingDataset(train_df, args.img_root)
    val_ds   = MultiCamDrivingDataset(val_df, args.img_root)
    test_ds  = MultiCamDrivingDataset(test_df, args.img_root)                                                   

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    trainer.check_sampler_balance(train_loader)

    # 4. Model & Optimizer
    # PRE 설정을 인자로 받거나 여기서 고정
    PRE = True 
    model = MultiCamParkingModel(pretrained=PRE, num_classes=cfg.NUM_CLASSES).to(device)
    
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    # Scheduler setup
    scheduler = None
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'cosine_restart':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,        # 첫 주기 (epoch)
            T_mult=2,      # 주기 증가 배수
            eta_min=1e-6   # min_lr
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # 5. ==================================================== Training Loop =======================================================
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    best_val_loss = float('inf')
    early_stop_cnt = 0

    print(f"\n🔥 Start Training...")
    for epoch in range(args.epochs):
        is_onecycle = (args.scheduler == 'onecycle')
        
        # trainer 모듈의 함수 사용
        train_loss, train_reg, train_cls = trainer.train_one_epoch(
            model, train_loader, criterion_reg, criterion_cls, optimizer, device, scaler, scheduler, is_onecycle
        )
        
        # [수정됨] 반환값 개수와 순서가 trainer.py와 일치해야 합니다.
        (val_loss, val_reg, val_cls,         # Loss 3개
         val_lin_mae, val_ang_mae,           # MAE 2개
         val_acc, val_class_accs,            # Acc 관련
         val_prec, val_recall, val_f1, val_cm, # Cls Metrics
         all_val_targets_lin, all_val_preds_lin, # Scatter용 데이터
         all_val_targets_ang, all_val_preds_ang) = trainer.eval_one_epoch(
            model, val_loader, criterion_reg, criterion_cls, device
        )

        if args.scheduler == 'cosine': scheduler.step()
        elif args.scheduler == 'plateau': scheduler.step(val_loss)

        current_lr = trainer.get_lr(optimizer)

        # ---------------- Tensorboard Logging ----------------
        # 1. [Total Loss]
        writer.add_scalars("Total_Loss", {"Train": train_loss, "Val": val_loss}, epoch)

        # 2. [Task Loss]
        # train.py 수정 제안
        writer.add_scalars("Loss/Reg", {"Train": train_reg, "Val": val_reg}, epoch)
        writer.add_scalars("Loss/Cls", {"Train": train_cls, "Val": val_cls}, epoch)

        # 3. [Validation Metrics]
        writer.add_scalar("Val_Metrics/MAE_Linear", val_lin_mae, epoch)
        writer.add_scalar("Val_Metrics/MAE_Angular", val_ang_mae, epoch)
        writer.add_scalar("Val_Metrics/Accuracy", val_acc, epoch)
        writer.add_scalar("Val_Metrics/Precision", val_prec, epoch)
        writer.add_scalar("Val_Metrics/Recall", val_recall, epoch)
        writer.add_scalar("Val_Metrics/F1_Score", val_f1, epoch)

        # 4. [Class Accuracy]
        writer.add_scalar("Val_Class_Acc/0_Drive", val_class_accs[0], epoch)
        writer.add_scalar("Val_Class_Acc/1_Parked", val_class_accs[1], epoch)

        # 5. [LR]
        writer.add_scalar("LR", current_lr, epoch)

        # 6. [Confusion Matrix]
        class_names = ["Drive", "Parked"] 
        cm_figure = trainer.plot_confusion_matrix(val_cm, class_names)
        writer.add_figure("Confusion Matrix", cm_figure, epoch)
        plt.close(cm_figure)

        # 7. [Linear, Angular 값 시각화]
        fig_lin = trainer.plot_regression_scatter(all_val_targets_lin, all_val_preds_lin, "Linear Output Analysis")
        writer.add_figure("Analysis/Linear_Scatter", fig_lin, epoch)
        plt.close(fig_lin)

        fig_ang = trainer.plot_regression_scatter(all_val_targets_ang, all_val_preds_ang, "Angular Output Analysis")
        writer.add_figure("Analysis/Angular_Scatter", fig_ang, epoch)
        plt.close(fig_ang)

        # 8. [Histogram]
        # 모델의 예측값 분포가 정답 분포와 비슷한 모양인지 확인
        writer.add_histogram("Dist/Linear_Pred", np.array(all_val_preds_lin), epoch)
        writer.add_histogram("Dist/Linear_GT",   np.array(all_val_targets_lin), epoch)

        # ---------------- Terminal Output ----------------
        print(f"Epoch [{epoch+1}/{args.epochs}] | LR: {current_lr:.8f} | Total Val Loss: {val_loss:.5f}")
        print(f"   >> [Reg] MAE Lin: {val_lin_mae:.4f}, Ang: {val_ang_mae:.4f}")
        print(f"   >> [Cls] Acc: {val_acc:.2f}%")
        print(f"   >> [Detail] Drive: {val_class_accs[0]:.2f}% | Parked: {val_class_accs[1]:.2f}%")
        print(f"   >> [Metrics] Pre: {val_prec:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"   >> Confusion Matrix:\n{val_cm}")
   
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
    # =================================================================================================================================================================================

    # ---------------------------------------------------------
    # 최종 테스트 단계 (Test Loop)
    # ---------------------------------------------------------
    print("\n🔍 Starting Final Test with the Best Model...")
    
    # 1. 저장된 최고의 모델 가중치 불러오기
    # 그냥 model을 쓰면 '마지막 에포크'의 가중치라서 성능이 가장 좋은 상태가 아닐 수 있습니다.
    best_model_path = os.path.join(args.out_dir, "best_model.pth")
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded weights from {best_model_path}")
    else:
        print("⚠️ Warning: Best model not found. Testing with final epoch weights.")

    # 2. 테스트 데이터셋으로 평가 실행
    # eval_one_epoch 함수를 그대로 재활용하면 됩니다.
    (test_loss, test_reg, test_cls, 
     test_lin_mae, test_ang_mae, 
     test_acc, test_class_accs, 
     test_prec, test_recall, test_f1, test_cm,
     _, _, _, _) = trainer.eval_one_epoch(model, test_loader, criterion_reg, criterion_cls, device)
    
    print("="*40)
    print(f"🏆 Final Test Result")
    print(f"Loss: {test_loss:.6f}")
    print(f"Reg MAE -> Linear: {test_lin_mae:.4f}, Angular: {test_ang_mae:.4f}")
    print(f"Acc -> {test_acc:.2f}%")
    print(f"   >> [Detail] Drive: {test_class_accs[0]:.2f}% | Parked: {test_class_accs[1]:.2f}%")
    print(f"   >> [Metrics] Pre: {test_prec:.4f} | Rec: {test_recall:.4f} | F1: {test_f1:.4f}")
    print(f"   >> Confusion Matrix:\n{test_cm}")
    print("="*40)
    
    print("🏁 Training & Testing Finished.")

if __name__ == "__main__":
    main()