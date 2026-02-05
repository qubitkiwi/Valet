import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

import io
from PIL import Image

import torch
from torch.utils.data import Dataset


# --- 함수 정의 (lr 계산, 전처리, 데이터셋 클래스) ---
def calculate_new_lr(old_lr, old_batch, new_batch, method='linear'):
    ratio = new_batch / old_batch
    if method == 'linear':
        return old_lr * ratio
    elif method == 'sqrt':
        return old_lr * math.sqrt(ratio)


def preprocess(image, CROP_HEIGHT):
    image = image[CROP_HEIGHT:480, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # YUV 변환
    image = cv2.resize(image, (200, 66))  # Resize
    return image


# --- PyTorch Dataset 정의 ---
class MultiTaskDataset(Dataset):
    def __init__(self, image_paths, controls, signs, root_dir, transform=None):
        self.image_paths = image_paths
        self.controls = controls
        self.signs = signs
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx].strip())
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # image = preprocess(image, CROP_HEIGHT=100)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        control_val = self.controls[idx]  # [linear_x, angular_z] (CSV order)
        control_label = torch.tensor(control_val, dtype=torch.float32)

        sign_val = self.signs[idx]
        sign_label = torch.tensor(sign_val, dtype=torch.long)

        return image, control_label, sign_label


def confusion_matrix_to_tensorboard_image(cm: torch.Tensor):
    """
    cm: [C, C] int tensor (true rows, pred cols)
    returns: CHW float tensor in [0,1] for TensorBoard
    """
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm.cpu().numpy(), interpolation="nearest")
    plt.title("Confusion Matrix (Val)")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()

    # write counts
    C = cm.shape[0]
    cm_np = cm.cpu().numpy()
    for i in range(C):
        for j in range(C):
            plt.text(j, i, str(cm_np[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    pil_img = Image.open(buf).convert("RGB")
    img = np.array(pil_img).astype(np.float32) / 255.0   # HWC
    img = torch.from_numpy(img).permute(2, 0, 1)         # CHW
    return img