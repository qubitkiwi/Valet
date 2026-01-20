#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
========
실물 로봇 주차 IL/DAgger 데이터셋으로 PilotNet6Ch 학습 (p1/p2/p3 one-hot condition 포함)

반영 사항
- aug_dataset 구조(run_xxx/<variant>/episode_yyy)에서 "같은 주행"의 증강본이 train/val로 갈라지는 누수 방지
  => run_name/episode_name 기준 Group Split
- CosineAnnealingLR 스케줄러 추가
- TensorBoard 로깅 추가
- tqdm 진행바 추가 (train/val 둘 다)
- 입력 사이즈 유지: (W,H) = (200,66)
- front: 위쪽 절반 제거 후 (200,66) 유지
- rear: crop 없이 (200,66)
- ✅ az 스케일링: az_train = az / steer_scale (기본 3.0)
- ✅ normalize 옵션: --normalize 켜면 ImageNet mean/std 적용 (train & val 동일)
"""

import csv
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.pilotnet6ch import PilotNet6Ch


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_episode_groups(dataset_dir: Path) -> Dict[str, List[Path]]:
    """
    aug_dataset 구조:
      run_000/orig/episode_000
      run_000/bri_0.5/episode_000
      ...

    같은 주행(run/episode)을 variant별로 묶어서 group split(누수 방지) 하기 위한 함수.
    key: "run_000/episode_000"
    val: [Path(.../orig/episode_000), Path(.../bri_0.5/episode_000), ...]
    """
    runs = sorted([p for p in dataset_dir.glob("run_*") if p.is_dir()])
    groups: Dict[str, List[Path]] = defaultdict(list)

    for run in runs:
        run_name = run.name
        for ep in sorted(run.glob("*/episode_*")):
            if not ep.is_dir():
                continue
            csv_path = ep / "actions.csv"
            front_dir = ep / "front"
            rear_dir = ep / "rear"
            if not (csv_path.exists() and front_dir.exists() and rear_dir.exists()):
                continue
            key = f"{run_name}/{ep.name}"  # ✅ variant 제외
            groups[key].append(ep)

    # variant 순서 고정(선택)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda p: p.parent.name)

    return dict(groups)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"p1", "p2", "p3", "front_image", "rear_image", "linear_x", "angular_z"}
        for r in reader:
            if not need.issubset(set(r.keys())):
                continue
            rows.append(r)
    return rows


def load_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def resize_bgr(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    if (img_bgr.shape[1] != w) or (img_bgr.shape[0] != h):
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr


def crop_front_bottom_half_keep_size(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    front 전용:
    - 먼저 (w,h)로 맞춘 뒤
    - 위쪽 절반을 잘라내고(아래 절반만)
    - 다시 (w,h)로 리사이즈해서 최종 입력 사이즈 유지
    """
    img_bgr = resize_bgr(img_bgr, w, h)           # (h,w,3)
    hh = img_bgr.shape[0]
    img_bgr = img_bgr[hh // 2:, :, :]             # 아래 절반
    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr


def bgr_to_chw_float(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img


# -----------------------------
# Dataset
# -----------------------------
class ParkingEpisodeDataset(Dataset):
    """
    반환:
      x: (6,H,W)
      p_oh: (3,)
      y: (2,)  where y = [linear_x, az_scaled]
    """
    def __init__(
        self,
        episode_dirs: List[Path],
        width: int,
        height: int,
        normalize: bool = False,
        steer_scale: float = 3.0,
    ):
        self.width = width
        self.height = height
        self.normalize = normalize
        self.steer_scale = float(steer_scale) if abs(float(steer_scale)) > 1e-9 else 1.0

        self.samples: List[Tuple[Path, Path, float, float, float, float, float]] = []

        for ep in episode_dirs:
            csv_path = ep / "actions.csv"
            rows = read_csv_rows(csv_path)
            for r in rows:
                try:
                    p1 = float(r["p1"])
                    p2 = float(r["p2"])
                    p3 = float(r["p3"])

                    front_rel = r["front_image"].strip()
                    rear_rel  = r["rear_image"].strip()

                    front_path = (ep / front_rel).resolve()
                    rear_path  = (ep / rear_rel).resolve()

                    lx = float(r["linear_x"])
                    az = float(r["angular_z"])

                    if front_path.exists() and rear_path.exists():
                        self.samples.append((front_path, rear_path, p1, p2, p3, lx, az))
                except Exception:
                    continue

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        front_path, rear_path, p1, p2, p3, lx, az = self.samples[idx]

        front = load_bgr(front_path)
        rear  = load_bgr(rear_path)
        if front is None or rear is None:
            j = (idx + 1) % len(self.samples)
            return self.__getitem__(j)

        # front: 위쪽 절반 제거 -> (W,H) 유지
        front = crop_front_bottom_half_keep_size(front, self.width, self.height)

        # rear: crop 없음
        rear = resize_bgr(rear, self.width, self.height)

        f = bgr_to_chw_float(front)
        r = bgr_to_chw_float(rear)

        if self.normalize:
            f = (f - self.mean) / self.std
            r = (r - self.mean) / self.std

        x6 = np.concatenate([f, r], axis=0)  # (6,H,W)
        x = torch.from_numpy(x6).float()

        # p1,p2,p3는 0/1 one-hot 그대로
        p_oh = torch.tensor([p1, p2, p3], dtype=torch.float32)

        # ✅ az 스케일링: 학습 타깃을 [-1,1] 근처로 맞춤
        az_scaled = float(az) / self.steer_scale
        y = torch.tensor([lx, az_scaled], dtype=torch.float32)

        return x, p_oh, y


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    epoch: int = 0
) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"val   E{epoch:03d}", leave=False)
    for x, p_oh, y in pbar:
        x = x.to(device, non_blocking=True)
        p_oh = p_oh.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x, p_oh)
                loss = loss_fn(pred, y)
        else:
            pred = model(x, p_oh)
            loss = loss_fn(pred, y)

        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs
        pbar.set_postfix(loss=float(loss.item()))

    return total / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    grad_clip: float = 0.0,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    log_every: int = 20,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    pbar = tqdm(loader, desc=f"train E{epoch:03d}", leave=False)
    for step, (x, p_oh, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        p_oh = p_oh.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x, p_oh)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x, p_oh)
            loss = loss_fn(pred, y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs

        pbar.set_postfix(loss=float(loss.item()))

        if writer is not None and (step % log_every == 0):
            global_step = epoch * len(loader) + step
            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            writer.add_scalar("lr_step", optimizer.param_groups[0]["lr"], global_step)
            writer.flush()

    return total / max(n, 1)


# -----------------------------
# Checkpoint
# -----------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args: dict,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    obj = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "args": args,
    }
    if scheduler is not None:
        obj["scheduler"] = scheduler.state_dict()
    torch.save(obj, str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", 1e18))
    return epoch, best_val


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="/home/sechankim/ros2_ws/src/parking_server/dataset/aug_dataset")
    ap.add_argument("--out_dir", type=str, default="./train_log")

    # ⚠️ 참고: action="store_true" + default=True면 항상 True라서 옵션 의미가 없어짐.
    # 네가 원래 의도대로 "기본 ON"이면 그냥 그대로 두고, 끄고 싶으면 코드 개선 필요.
    ap.add_argument("--early_stop", action="store_true", default=True)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_delta", type=float, default=1e-5)

    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=66)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # ✅ az 스케일링 (steer input scale)
    ap.add_argument("--steer_scale", type=float, default=3.0, help="az_train = az / steer_scale")

    # ✅ CosineAnnealingLR
    ap.add_argument("--sched_tmax", type=int, default=200, help="CosineAnnealingLR T_max (보통 epochs와 동일)")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="CosineAnnealingLR eta_min")

    # ✅ TensorBoard
    ap.add_argument("--tb", action="store_true", default=True, help="TensorBoard logging")
    ap.add_argument("--tb_log_every", type=int, default=20, help="train_step 로깅 주기(step)")

    ap.add_argument("--resume", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    ensure_dir(out_root)
    existing = sorted([p for p in out_root.glob("exp_*") if p.is_dir()])
    exp_id = len(existing)
    exp_dir = out_root / f"exp_{exp_id:03d}"
    ensure_dir(exp_dir)

    ckpt_dir = exp_dir / "ckpt"
    ensure_dir(ckpt_dir)

    # -----------------------------
    # ✅ Group Split (누수 방지)
    # -----------------------------
    groups = list_episode_groups(dataset_dir)
    keys = list(groups.keys())
    if len(keys) == 0:
        raise RuntimeError(f"No episode groups found under: {dataset_dir}/run_*/<variant>/episode_*")

    rng = random.Random(args.seed)
    rng.shuffle(keys)

    n_val = max(1, int(len(keys) * args.val_ratio))
    val_keys = keys[:n_val]
    tr_keys = keys[n_val:]

    val_eps = [ep for k in val_keys for ep in groups[k]]
    tr_eps = [ep for k in tr_keys for ep in groups[k]]

    print(f"[INFO] dataset_dir: {dataset_dir}")
    print(f"[INFO] groups total={len(keys)} train_groups={len(tr_keys)} val_groups={len(val_keys)}")
    print(f"[INFO] episodes total={len(tr_eps) + len(val_eps)} train_eps={len(tr_eps)} val_eps={len(val_eps)}")
    print(f"[INFO] steer_scale: {args.steer_scale}")
    print(f"[INFO] normalize: {args.normalize}")
    print(f"[INFO] exp_dir: {exp_dir}")

    train_ds = ParkingEpisodeDataset(tr_eps, args.width, args.height, normalize=args.normalize, steer_scale=args.steer_scale)
    val_ds   = ParkingEpisodeDataset(val_eps, args.width, args.height, normalize=args.normalize, steer_scale=args.steer_scale)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset after parsing. train={len(train_ds)} val={len(val_ds)}")

    if args.num_workers is None:
        args.num_workers = 0
    args.num_workers = int(args.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = PilotNet6Ch().to(device)

    # 더미 forward 1회
    model.eval()
    with torch.no_grad():
        dummy_x = torch.zeros(1, 6, args.height, args.width, device=device)
        dummy_p = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        _ = model(dummy_x, dummy_p)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ✅ CosineAnnealingLR (epoch 단위 step)
    tmax = int(args.sched_tmax) if args.sched_tmax > 0 else int(args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tmax,
        eta_min=float(args.min_lr),
    )

    # ✅ TensorBoard
    writer = None
    if args.tb:
        writer = SummaryWriter(log_dir=str(exp_dir / "tb"))
        writer.add_text("info/exp_dir", str(exp_dir), 0)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], 0)
        writer.add_text("info/steer_scale", str(args.steer_scale), 0)
        writer.add_text("info/normalize", str(args.normalize), 0)
        writer.flush()

    start_epoch = 0
    best_val = 1e18
    bad_epochs = 0

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume not found: {resume_path}")
        start_epoch, best_val = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"[INFO] resumed from {resume_path} @ epoch={start_epoch} best_val={best_val:.6f}")

    log_path = exp_dir / "logs.csv"
    if not log_path.exists() or start_epoch == 0:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "lr"])

    (exp_dir / "args.txt").write_text("\n".join([f"{k}={v}" for k, v in vars(args).items()]), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp=args.amp,
            grad_clip=args.grad_clip,
            writer=writer,
            epoch=epoch,
            log_every=int(args.tb_log_every),
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            amp=args.amp,
            epoch=epoch,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch=epoch,
            best_val=best_val,
            args=vars(args),
            scheduler=scheduler
        )

        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            bad_epochs = 0
            best_path = exp_dir / "best.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                best_val=best_val,
                args=vars(args),
                scheduler=scheduler
            )
        else:
            bad_epochs += 1

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", lr_now, epoch)
            writer.flush()

        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr_now:.8e}"])

        dt = time.time() - t0
        print(
            f"[E{epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} "
            f"{'(BEST)' if improved else ''} time={dt:.1f}s lr={lr_now:.2e} ckpt={ckpt_path.name}"
        )

        if args.early_stop and (bad_epochs >= args.patience):
            print(f"[EARLY STOP] no improvement for {bad_epochs} epochs (patience={args.patience}), stopping.")
            break

    if writer is not None:
        writer.close()

    print(f"[DONE] best_val={best_val:.6f} @ {exp_dir}")


if __name__ == "__main__":
    main()
