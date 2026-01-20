#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np

HEADER = ["p1", "p2", "p3", "front_image", "rear_image", "linear_x", "angular_z"]

BLUR_PRESETS = [
    ("blur_k3_sig0.3", (3, 3), 0.3),
    ("blur_k3_sig0.6", (3, 3), 0.6),
    ("blur_k5_sig1.0", (5, 5), 1.0),
    ("blur_k5_sig1.5", (5, 5), 1.5),
    ("blur_k7_sig2.0", (7, 7), 2.0),
    ("blur_k7_sig3.0", (7, 7), 3.0),
]

# -----------------------------
# IO helpers
# -----------------------------
def list_episode_dirs(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.glob("episode_*") if p.is_dir()])


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if all(k in r for k in HEADER):
                rows.append(r)
    return rows


def write_rows(csv_path: Path, rows: List[Dict[str, str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for rr in rows:
            w.writerow(rr)


def imread_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def imwrite_bgr(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    return ok


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


# -----------------------------
# Augment ops
# -----------------------------
def aug_brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def aug_contrast(img, factor):
    x = img.astype(np.float32)
    return np.clip((x - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)


def aug_gamma(img, gamma):
    inv = 1.0 / gamma
    table = ((np.linspace(0, 1, 256) ** inv) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def aug_blur(img, k, sigma):
    return cv2.GaussianBlur(img, k, sigmaX=sigma, sigmaY=sigma)


# -----------------------------
# Episode processing
# -----------------------------
def process_episode(ep_in: Path, ep_out: Path, mode: str) -> int:
    """
    returns: number of kept rows
    """
    rows = read_rows(ep_in / "actions.csv")
    out_rows = []

    for r in rows:
        f_rel, r_rel = r["front_image"], r["rear_image"]
        f_src = (ep_in / f_rel)
        r_src = (ep_in / r_rel)

        # orig는 "복사"로 처리(빠르고 손실 없음)
        if mode == "orig":
            if not f_src.exists() or not r_src.exists():
                continue
            safe_copy(f_src, ep_out / f_rel)
            safe_copy(r_src, ep_out / r_rel)
            out_rows.append(r)
            continue

        # 나머지는 decode -> augment -> encode
        f_img = imread_bgr(f_src)
        r_img = imread_bgr(r_src)
        if f_img is None or r_img is None:
            continue

        f2, r2 = f_img, r_img

        if mode == "bri_0.5":
            f2, r2 = aug_brightness(f2, 0.5), aug_brightness(r2, 0.5)
        elif mode == "bri_1.5":
            f2, r2 = aug_brightness(f2, 1.5), aug_brightness(r2, 1.5)
        elif mode == "con_0.5":
            f2, r2 = aug_contrast(f2, 0.5), aug_contrast(r2, 0.5)
        elif mode == "gam_1.2":
            f2, r2 = aug_gamma(f2, 1.2), aug_gamma(r2, 1.2)
        elif mode == "gam_1.4":
            f2, r2 = aug_gamma(f2, 1.4), aug_gamma(r2, 1.4)
        elif mode.startswith("blur"):
            for name, k, sig in BLUR_PRESETS:
                if mode == name:
                    f2, r2 = aug_blur(f2, k, sig), aug_blur(r2, k, sig)
                    break

        ok1 = imwrite_bgr(ep_out / f_rel, f2)
        ok2 = imwrite_bgr(ep_out / r_rel, r2)
        if ok1 and ok2:
            out_rows.append(r)

    write_rows(ep_out / "actions.csv", out_rows)
    return len(out_rows)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default='/home/sechankim/ros2_ws/src/parking_server/dataset/preprocessed_dataset')
    ap.add_argument("--out_dir", default='/home/sechankim/ros2_ws/src/parking_server/dataset/aug_dataset')

    ap.add_argument("--skip_existing", action="store_true", default=False,
                    help="(기본 True) out_dir에 episode/actions.csv가 있으면 해당 episode는 스킵")

    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="특정 mode 폴더를 통째로 지우고 다시 생성(재증강)")
    ##평소엔 --skip_existing 없이 실행(=전부 생성)
###이미 만든 걸 유지하고 싶을 때만 --skip_existing 켜기

    ap.add_argument("--modes", nargs="*", default=None,
                    help="특정 mode만 돌리고 싶을 때. 예: --modes orig bri_0.5 blur_k5_sig1.0")

    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    all_modes = (
        ["orig", "bri_0.5", "bri_1.5", "con_0.5", "gam_1.2", "gam_1.4"]
        + [name for name, _, _ in BLUR_PRESETS]
    )
    modes = args.modes if args.modes else all_modes

    # overwrite_mode 옵션이면 mode 디렉토리만 삭제
    if args.overwrite_mode:
        for run in sorted(in_dir.glob("run_*")):
            if not run.is_dir():
                continue
            for mode in modes:
                mode_dir = out_dir / run.name / mode
                if mode_dir.exists():
                    shutil.rmtree(mode_dir)

    for run in sorted(in_dir.glob("run_*")):
        if not run.is_dir():
            continue

        print(f"[RUN] {run.name}")
        for mode in modes:
            mode_dir = out_dir / run.name / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            for ep in list_episode_dirs(run):
                ep_out = mode_dir / ep.name

                # ✅ 증분 처리: 이미 actions.csv가 있으면 스킵
                if args.skip_existing and (ep_out / "actions.csv").exists():
                    continue

                (ep_out / "front").mkdir(parents=True, exist_ok=True)
                (ep_out / "rear").mkdir(parents=True, exist_ok=True)

                kept = process_episode(ep, ep_out, mode)
                # 원하면 로그
                # print(f"  [{mode}] {ep.name}: kept={kept}")

    print(f"[DONE] aug_dataset created at {out_dir}")


if __name__ == "__main__":
    main()
