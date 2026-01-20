#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple
import os

import cv2
import numpy as np
import torch

from models.pilotnet6ch import PilotNet6Ch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 학습 입력 크기와 반드시 동일
IMG_W = int(os.environ.get("PARKING_IMG_W", "200"))
IMG_H = int(os.environ.get("PARKING_IMG_H", "66"))

CKPT_PATH = Path(os.environ.get(
    "PARKING_CKPT",
    "/home/sechankim/ros2_ws/src/parking_server/train_log/exp_009/best.pt"
))

# ✅ az 스케일(학습에서 az_train = az / steer_scale 했으므로, 추론에서 다시 * steer_scale)
STEER_SCALE = float(os.environ.get("PARKING_STEER_SCALE", "3.0"))
if abs(STEER_SCALE) < 1e-9:
    STEER_SCALE = 1.0


def slot_to_onehot(slot: str) -> np.ndarray:
    # ✅ 요청대로 p1/p2/p3만 원핫 처리 (1,2,3 변환 안 함)
    s = (slot or "").lower().strip()
    if s == "p1":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if s == "p2":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if s == "p3":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def decode_jpeg_to_bgr(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img


def resize_bgr(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr


def crop_front_bottom_half_keep_size(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    ✅ train.py와 동일한 front crop:
    - (w,h) resize
    - 위 절반 제거(아래 절반만)
    - 다시 (w,h) resize
    """
    img_bgr = resize_bgr(img_bgr, w, h)
    hh = img_bgr.shape[0]
    img_bgr = img_bgr[hh // 2:, :, :]
    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr


def bgr_to_chw_rgb_float(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


def load_model_and_flags(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model = PilotNet6Ch().to(DEVICE)
    model.eval()

    # ✅ 더미 forward로 lazy fc 초기화
    with torch.no_grad():
        dummy_x = torch.zeros(1, 6, IMG_H, IMG_W, device=DEVICE)
        dummy_p = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
        _ = model(dummy_x, dummy_p)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ✅ train args에서 normalize/steer_scale 읽기 (있으면)
    normalize = False
    steer_scale = STEER_SCALE
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        normalize = bool(ckpt["args"].get("normalize", False))
        steer_scale = float(ckpt["args"].get("steer_scale", steer_scale))

    # env로 강제 override도 가능
    env_norm = os.environ.get("PARKING_NORMALIZE", "").strip()
    if env_norm in ("1", "true", "TRUE", "yes", "YES"):
        normalize = True
    if env_norm in ("0", "false", "FALSE", "no", "NO"):
        normalize = False

    return model, normalize, steer_scale


MODEL, USE_NORMALIZE, CKPT_STEER_SCALE = load_model_and_flags(CKPT_PATH)

# ImageNet mean/std (train.py와 동일)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def infer(front_bytes: bytes, rear_bytes: bytes, slot: str) -> Tuple[float, float]:
    # decode
    f_bgr = decode_jpeg_to_bgr(front_bytes)
    r_bgr = decode_jpeg_to_bgr(rear_bytes)

    # ✅ train과 동일: front만 crop 적용
    f_bgr = crop_front_bottom_half_keep_size(f_bgr, IMG_W, IMG_H)
    # rear는 crop 없음, resize만
    r_bgr = resize_bgr(r_bgr, IMG_W, IMG_H)

    # to CHW float
    f = bgr_to_chw_rgb_float(f_bgr)
    r = bgr_to_chw_rgb_float(r_bgr)

    # ✅ normalize 옵션
    if USE_NORMALIZE:
        f = (f - MEAN) / STD
        r = (r - MEAN) / STD

    x6 = np.concatenate([f, r], axis=0).astype(np.float32)  # (6,H,W)
    p = slot_to_onehot(slot).astype(np.float32)

    x = torch.from_numpy(x6).unsqueeze(0).to(DEVICE)
    p_oh = torch.from_numpy(p).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = MODEL(x, p_oh)[0].detach().cpu().numpy()

    # out[0] = linear_x (그대로)
    # out[1] = az_scaled (학습에서 /steer_scale 했으므로 복원)
    lx = float(out[0])
    az = float(out[1]) * float(CKPT_STEER_SCALE)

    return lx, az
