import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

# ===============================================================================================================
def eval_one_epoch(model, loader, criterion_reg, criterion_cls, device):
    model.eval()

    total_loss_sum = 0.0
    total_reg_loss_sum = 0.0  # [추가] Reg Loss 따로 집계
    total_cls_loss_sum = 0.0  # [추가] Cls Loss 따로 집계
    
    abs_err_lin = 0.0
    abs_err_ang = 0.0
    
    # [추가] Scatter Plot을 위한 리스트들
    all_preds_lin = []
    all_targets_lin = []
    all_preds_ang = []
    all_targets_ang = []
    
    # Classification용
    all_preds_cls = []
    all_targets_cls = []
    
    n = 0

    with torch.no_grad():
        for images, reg_labels, cls_labels in loader:
            images = images.to(device, non_blocking=True)
            reg_labels = reg_labels.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)

            outputs = model(images)
            pred_reg = outputs['control']
            pred_cls = outputs['class']
            
            # Loss 계산
            loss_reg = criterion_reg(pred_reg, reg_labels)
            loss_cls = criterion_cls(pred_cls, cls_labels)
            loss = loss_reg + loss_cls
            
            bs = reg_labels.size(0)
            n += bs
            
            # Sum Updates
            total_loss_sum += loss.item() * bs
            total_reg_loss_sum += loss_reg.item() * bs # [추가]
            total_cls_loss_sum += loss_cls.item() * bs # [추가]

            # MAE Calculation
            abs_err_lin += torch.abs(pred_reg[:, 0] - reg_labels[:, 0]).sum().item()
            abs_err_ang += torch.abs(pred_reg[:, 1] - reg_labels[:, 1]).sum().item()
            
            # --- 데이터 수집 (CPU로 이동) ---
            # 1. Regression (Linear & Angular)
            all_preds_lin.extend(pred_reg[:, 0].cpu().numpy())
            all_targets_lin.extend(reg_labels[:, 0].cpu().numpy())
            
            all_preds_ang.extend(pred_reg[:, 1].cpu().numpy())
            all_targets_ang.extend(reg_labels[:, 1].cpu().numpy())

            # 2. Classification
            _, predicted_class = torch.max(pred_cls, 1)
            all_preds_cls.extend(predicted_class.cpu().numpy())
            all_targets_cls.extend(cls_labels.cpu().numpy())

    # 평균 계산
    avg_loss = total_loss_sum / n
    avg_reg_loss = total_reg_loss_sum / n  # [추가]
    avg_cls_loss = total_cls_loss_sum / n  # [추가]
    
    lin_mae  = abs_err_lin / n
    ang_mae  = abs_err_ang / n

    # Metrics
    accuracy = np.mean(np.array(all_preds_cls) == np.array(all_targets_cls)) * 100.0
    precision = precision_score(all_targets_cls, all_preds_cls, average='macro', zero_division=0)
    recall = recall_score(all_targets_cls, all_preds_cls, average='macro', zero_division=0)
    f1 = f1_score(all_targets_cls, all_preds_cls, average='macro', zero_division=0)
    
    conf_matrix = confusion_matrix(all_targets_cls, all_preds_cls)
    
    class_accs = (conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-6)) * 100.0
    class_accs = class_accs.tolist()

    # [반환값 대폭 추가]
    # Loss 3종류, MAE 2종류, Cls지표들, 그리고 Scatter용 리스트들
    return (avg_loss, avg_reg_loss, avg_cls_loss, 
            lin_mae, ang_mae, 
            accuracy, class_accs, precision, recall, f1, conf_matrix,
            all_targets_lin, all_preds_lin, all_targets_ang, all_preds_ang)

# ===============================================================================================================

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, loader, criterion_reg, criterion_cls, optimizer, device, scaler, scheduler=None, is_onecycle=False):
    model.train()
    total_loss_sum = 0.0
    reg_loss_sum = 0.0
    cls_loss_sum = 0.0
    
    for images, reg_labels, cls_labels in loader:
        images = images.to(device, non_blocking=True)
        reg_labels = reg_labels.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images)
            pred_reg = outputs['control']
            pred_cls = outputs['class']

            loss_reg = criterion_reg(pred_reg, reg_labels)
            loss_cls = criterion_cls(pred_cls, cls_labels)
            loss = loss_reg + loss_cls

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        scaler.step(optimizer)
        scaler.update()

        if is_onecycle and scheduler:
            scheduler.step()

        bs = images.size(0)
        total_loss_sum += loss.item() * bs
        reg_loss_sum += loss_reg.item() * bs
        cls_loss_sum += loss_cls.item() * bs

    dataset_len = len(loader.dataset)
    return (total_loss_sum / dataset_len), (reg_loss_sum / dataset_len), (cls_loss_sum / dataset_len)

def check_sampler_balance(loader):
    """
    WeightedRandomSampler가 잘 적용되었는지 첫 배치만 확인하는 함수
    """
    print("\n🔍 [Check] Verifying WeightedRandomSampler...")
    try:
        # 첫 번째 배치만 가져와봄 (DataLoader가 iterable이므로 iter() 사용)
        temp_batch = next(iter(loader))
        _, _, temp_cls_labels = temp_batch
        
        temp_labels = temp_cls_labels.numpy()
        unique, counts = np.unique(temp_labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        
        print(f"   >> Batch Size: {len(temp_labels)}")
        print(f"   >> Class Counts in Batch: {count_dict}")
        
        total = len(temp_labels)
        ratio_0 = count_dict.get(0, 0) / total * 100
        ratio_1 = count_dict.get(1, 0) / total * 100
        print(f"   >> Ratio -> Drive(0): {ratio_0:.1f}% | Stop(1): {ratio_1:.1f}%")

        if abs(ratio_0 - ratio_1) < 20: # 차이가 20%p 이내면 균형 잡힌 걸로 간주
            print("   ✅ Sampler seems to be WORKING! (Balanced)")
        else:
            print("   ⚠️ WARNING: NOT balanced. Check Sampler code.")
    except Exception as e:
        print(f"   ⚠️ Error checking sampler: {e}")
    print("=" * 60)

def plot_confusion_matrix(cm, class_names):
    """
    Confusion Matrix(numpy array)를 받아서 matplotlib Figure 객체로 변환
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 매트릭스 안에 숫자 텍스트 넣기 (배경색에 따라 글자색 변경)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, format(cm[i, j], 'd'), 
                     horizontalalignment="center", color=color)
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return figure

def plot_regression_scatter(targets, preds, title="Regression Analysis"):
    """
    x축: 정답(Ground Truth), y축: 예측(Prediction)
    이상적인 경우 y=x 선 위에 점들이 모여야 함.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.5, s=10)
    
    # y=x 기준선 (완벽한 예측선)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    ax.set_title(title)
    ax.grid(True)
    return fig