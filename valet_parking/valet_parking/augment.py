#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PATH_HEADER = [
    "path",
    "front_cam", "rear_cam", "left_cam", "right_cam",
    "linear_x", "angular_z",
]

BLUR_PRESETS = [
    ("blur_k3_sig0.3", (3, 3), 0.3),
    ("blur_k3_sig0.6", (3, 3), 0.6),
    ("blur_k5_sig1.0", (5, 5), 1.0),
    ("blur_k5_sig1.5", (5, 5), 1.5),
    ("blur_k7_sig2.0", (7, 7), 2.0),
    ("blur_k7_sig3.0", (7, 7), 3.0),
]

ALL_MODES = (
    ["orig", "bri_0.5", "bri_1.5", "con_0.5", "gam_1.2", "gam_1.4"]
    + [name for name, _, _ in BLUR_PRESETS]
)

CAM_KEYS = ["front_cam", "rear_cam", "left_cam", "right_cam"]


def ensure_rel(p: str) -> str:
    p = (p or "").strip()
    return p[1:] if p.startswith("/") else p


def parse_run_episode(path_str: str) -> Tuple[str, str]:
    parts = [x for x in (path_str or "").split("/") if x]
    run_name = None
    ep_name = None
    for x in parts:
        if x.startswith("run_"):
            run_name = x
        if x.startswith("episode_"):
            ep_name = x
    if run_name is None or ep_name is None:
        raise ValueError(f"bad path='{path_str}' (need run_*/episode_*)")
    return run_name, ep_name


def imread_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def imwrite_jpeg(path: Path, img: np.ndarray, quality: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]))


def safe_copy(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception:
        return False


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


def apply_mode(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "bri_0.5":
        return aug_brightness(img, 0.5)
    if mode == "bri_1.5":
        return aug_brightness(img, 1.5)
    if mode == "con_0.5":
        return aug_contrast(img, 0.5)
    if mode == "gam_1.2":
        return aug_gamma(img, 1.2)
    if mode == "gam_1.4":
        return aug_gamma(img, 1.4)
    if mode.startswith("blur"):
        for name, k, sig in BLUR_PRESETS:
            if mode == name:
                return aug_blur(img, k, sig)
    return img


def read_path_csv(path_csv: Path) -> List[Dict[str, str]]:
    rows = []
    with path_csv.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != PATH_HEADER:
            raise RuntimeError(
                f"[BAD HEADER] {path_csv}\n expected={PATH_HEADER}\n got={rd.fieldnames}"
            )
        for r in rd:
            rows.append({
                "path": (r["path"] or "").strip(),
                "front_cam": ensure_rel(r["front_cam"]),
                "rear_cam":  ensure_rel(r["rear_cam"]),
                "left_cam":  ensure_rel(r["left_cam"]),
                "right_cam": ensure_rel(r["right_cam"]),
                "linear_x": (r["linear_x"] or "").strip(),
                "angular_z": (r["angular_z"] or "").strip(),
            })
    return rows


def write_path_csv(csv_path: Path, rows: List[Dict[str, str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=PATH_HEADER)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in PATH_HEADER})
    tmp.replace(csv_path)


def process_row(dataset_root: Path, out_ep_dir: Path, row: Dict[str, str],
                mode: str, jpeg_quality: int) -> bool:
    base_in = dataset_root / row["path"]
    srcs = {k: (base_in / row[k]) for k in CAM_KEYS}
    for p in srcs.values():
        if not p.exists():
            return False

    dsts = {k: (out_ep_dir / row[k]) for k in CAM_KEYS}

    if mode == "orig":
        for k in CAM_KEYS:
            if not safe_copy(srcs[k], dsts[k]):
                return False
        return True

    for k in CAM_KEYS:
        img = imread_bgr(srcs[k])
        if img is None:
            return False
        img2 = apply_mode(img, mode)
        if not imwrite_jpeg(dsts[k], img2, jpeg_quality):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/home/sechankim/ros2_ws/src/dataset/aug_valet_parking")
    ap.add_argument("--input_name", default="total_actions_cleaned.csv",
                    help="각 run 폴더에 있는 path 헤더 CSV 파일명")
    ap.add_argument("--augment_dirname", default="augment_dataset")

    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--skip_existing", action="store_true", default=False)

    ap.add_argument("--jpeg_quality", type=int, default=50)
    ap.add_argument("--global_csv", default="global_total_actions_path.csv")

    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    aug_root = dataset_root / args.augment_dirname
    aug_root.mkdir(parents=True, exist_ok=True)

    modes = args.modes if args.modes else list(ALL_MODES)

    run_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not run_dirs:
        print(f"[ERR] no run_* under {dataset_root}")
        return

    if args.overwrite:
        for run_dir in run_dirs:
            for mode in modes:
                mode_dir = aug_root / run_dir.name / mode
                if mode_dir.exists():
                    shutil.rmtree(mode_dir)

    # ✅ global rows: augment ONLY
    global_rows: List[Dict[str, str]] = []

    # tqdm total = sum(rows) * len(modes)
    per_run_rows = {}
    total_work = 0
    for run_dir in run_dirs:
        in_csv = run_dir / args.input_name
        if not in_csv.exists():
            continue
        rows = read_path_csv(in_csv)
        per_run_rows[run_dir.name] = rows
        total_work += len(rows) * len(modes)

    if total_work == 0:
        print("[ERR] nothing to process (no csv or empty rows)")
        return

    pbar = tqdm(total=total_work, desc="augment", unit="row")

    try:
        for run_dir in run_dirs:
            if run_dir.name not in per_run_rows:
                print(f"[SKIP] missing input csv: {run_dir / args.input_name}")
                continue

            base_rows = per_run_rows[run_dir.name]

            for mode in modes:
                mode_dir = aug_root / run_dir.name / mode
                mode_dir.mkdir(parents=True, exist_ok=True)

                mode_total: List[Dict[str, str]] = []

                for r in base_rows:
                    _, ep_name = parse_run_episode(r["path"])
                    out_ep_dir = mode_dir / ep_name

                    # 증분 스킵
                    if args.skip_existing and out_ep_dir.exists():
                        pbar.set_postfix_str(f"{run_dir.name}/{mode} (skip)")
                        pbar.update(1)
                        continue

                    ok = process_row(dataset_root, out_ep_dir, r, mode, args.jpeg_quality)

                    if ok:
                        out_path = f"{args.augment_dirname}/{run_dir.name}/{mode}/{ep_name}"
                        out_row = {
                            "path": out_path,
                            "front_cam": r["front_cam"],
                            "rear_cam":  r["rear_cam"],
                            "left_cam":  r["left_cam"],
                            "right_cam": r["right_cam"],
                            "linear_x":  r["linear_x"],
                            "angular_z": r["angular_z"],
                        }
                        mode_total.append(out_row)
                        global_rows.append(out_row)

                    pbar.set_postfix_str(f"{run_dir.name}/{mode}")
                    pbar.update(1)

                # mode별 total csv
                write_path_csv(mode_dir / "total_actions_path.csv", mode_total)

        # ✅ global csv: augment only
        write_path_csv(dataset_root / args.global_csv, global_rows)

    finally:
        pbar.close()

    print(f"\n[DONE] global csv (augment only): {dataset_root / args.global_csv}")
    print(f"       augment root: {aug_root}")


if __name__ == "__main__":
    main()
