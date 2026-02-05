#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Iterable
import cv2
import numpy as np

# =========================
# Paths (✅ BASE만 바꿈)
# =========================
BASE = Path("/home/elicer/jun_ws/")

# =========================
# data 이름 지정
# =========================
DATA_NAME = "total_data_final"


ORIG_DIR = BASE / "data" / DATA_NAME
ORIG_IMG_DIR = ORIG_DIR / "images"

AUG_ROOT = BASE / "data_augmented" / DATA_NAME  # ✅ 여기로만 저장

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# Aug defs
# =========================
def aug_dark(img: np.ndarray, factor: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def aug_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    # alpha < 1: 대비 감소, alpha > 1: 대비 증가
    img_f = img.astype(np.float32)
    out = 128.0 + alpha * (img_f - 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def aug_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    table = (np.arange(256) / 255.0) ** gamma
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

    return cv2.LUT(img, table)

def aug_weak_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
def aug_hflip(img: np.ndarray) -> np.ndarray:

    return cv2.flip(img, 1)




# =========================
# CSV I/O (timestamp, image_path, linear_x, angular_z, sign_class)
# =========================
def read_labels_csv(csv_path: Path) -> List[Tuple[str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {'timestamp', 'image_path', 'linear_x', 'angular_z', 'sign_class'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path.name} must have columns {required}, got {reader.fieldnames}")
        for r in reader:
            # keep numerical values as strings to avoid accidental reformatting
            rows.append((r["timestamp"], r["image_path"], r["linear_x"], r["angular_z"], r["sign_class"]))
    return rows

def write_labels_csv(csv_path: Path, rows: List[Tuple[str, str, str, str, str]]):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'image_path', 'linear_x', 'angular_z', 'sign_class'])
        for ts, img_path, lx, az, sc in rows:
            # write numeric fields verbatim as they came in (strings) to avoid reformatting
            w.writerow([ts, img_path, lx, az, sc])

def invert_angular_z_csv(in_csv: Path, out_csv: Path):
    rows = read_labels_csv(in_csv)
    flipped: List[Tuple[str, str, str, str, str]] = []
    for (ts, img, lx, az, sc) in rows:
        az_s = az.strip()
        # flip sign textually to avoid any numeric reformatting
        if az_s.startswith('-'):
            new_az = az_s[1:]
        elif az_s.startswith('+'):
            new_az = '-' + az_s[1:]
        else:
            new_az = '-' + az_s
        flipped.append((ts, img, lx, new_az, sc))
    write_labels_csv(out_csv, flipped)

# =========================1
# Utils
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def list_csvs(root: Path) -> List[Path]:
    return [p for p in root.glob("*.csv") if p.is_file()]

def make_aug_set(aug_name: str) -> Tuple[Path, Path]:
    aug_dir = AUG_ROOT / aug_name
    aug_images = aug_dir / "images"
    ensure_dir(aug_images)
    return aug_dir, aug_images

def copy_csvs(aug_dir: Path, orig_csvs: List[Path]):
    for c in orig_csvs:
        shutil.copy2(c, aug_dir / c.name)

# =========================
# Pipelines
# =========================
def run_dark_versions(factors: Iterable[float]):
    orig_imgs = list_images(ORIG_IMG_DIR)
    orig_csvs = list_csvs(ORIG_DIR)
    ensure_dir(AUG_ROOT)

    for f in factors:

        aug_name = f"dark_{f:.3f}"
        aug_dir, aug_img_dir = make_aug_set(aug_name)

        copy_csvs(aug_dir, orig_csvs)

        for img_path in orig_imgs:
            rel = img_path.relative_to(ORIG_IMG_DIR)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print("[WARN] unreadable:", img_path)
                continue

            out_path = aug_img_dir / rel
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), aug_dark(img, f))

        print(f"[DONE] {aug_name} -> {aug_dir}")


def run_contrast_versions(alphas: Iterable[float], beta: float = 0.0):
    """
    각 alpha 별로 폴더를 따로 만들어 저장:
      data_augmented/contrast_a0.800/images/...
      data_augmented/contrast_a1.100/images/...
    """
    orig_imgs = list_images(ORIG_IMG_DIR)
    orig_csvs = list_csvs(ORIG_DIR)
    ensure_dir(AUG_ROOT)

    for a in alphas:
        if a <= 0:
            raise ValueError(f"contrast alpha must be > 0, got {a}")

        aug_name = f"contrast_a{a:.3f}"
        aug_dir, aug_img_dir = make_aug_set(aug_name)

        # 라벨 변화 없음 → csv 복사
        copy_csvs(aug_dir, orig_csvs)

        for img_path in orig_imgs:
            rel = img_path.relative_to(ORIG_IMG_DIR)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print("[WARN] unreadable:", img_path)
                continue

            out_path = aug_img_dir / rel
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), aug_contrast(img, alpha=a))

        print(f"[DONE] {aug_name} -> {aug_dir}")


def run_gamma_versions(gammas: Iterable[float]):
    """
    각 gamma 별로 폴더를 따로 만들어 저장:
      data_augmented/gamma_g1.200/images/...
      data_augmented/gamma_g1.600/images/...
    """
    orig_imgs = list_images(ORIG_IMG_DIR)
    orig_csvs = list_csvs(ORIG_DIR)
    ensure_dir(AUG_ROOT)

    for g in gammas:
        if g <= 0:
            raise ValueError(f"gamma must be > 0, got {g}")

        aug_name = f"gamma_g{g:.3f}"
        aug_dir, aug_img_dir = make_aug_set(aug_name)

        # 라벨 변화 없음 → csv 복사
        copy_csvs(aug_dir, orig_csvs)

        for img_path in orig_imgs:
            rel = img_path.relative_to(ORIG_IMG_DIR)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print("[WARN] unreadable:", img_path)
                continue

            out_path = aug_img_dir / rel
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), aug_gamma(img, gamma=g))

        print(f"[DONE] {aug_name} -> {aug_dir}")


def run_blur_versions(ksize: int, sigmas: Iterable[float]):
    orig_imgs = list_images(ORIG_IMG_DIR)
    orig_csvs = list_csvs(ORIG_DIR)
    ensure_dir(AUG_ROOT)

    for s in sigmas:
        if s <= 0:
            raise ValueError(f"blur sigma must be > 0, got {s}")

        aug_name = f"blur_k{ksize}_sig{s:.3f}"
        aug_dir, aug_img_dir = make_aug_set(aug_name)

        copy_csvs(aug_dir, orig_csvs)

        for img_path in orig_imgs:
            rel = img_path.relative_to(ORIG_IMG_DIR)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print("[WARN] unreadable:", img_path)
                continue

            out_path = aug_img_dir / rel
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), aug_weak_blur(img, ksize, s))

        print(f"[DONE] {aug_name} -> {aug_dir}")


def run_hflip():
    orig_imgs = list_images(ORIG_IMG_DIR)
    orig_csvs = list_csvs(ORIG_DIR)
    ensure_dir(AUG_ROOT)

    aug_name = "hflip"
    aug_dir, aug_img_dir = make_aug_set(aug_name)

    for img_path in orig_imgs:
        rel = img_path.relative_to(ORIG_IMG_DIR)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] unreadable:", img_path)
            continue

        out_path = aug_img_dir / rel
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), aug_hflip(img))

    # Prefer specific CSV: total_data_A2B.csv, fallback to inverting all CSVs
    preferred = ORIG_DIR / "total_data_A2B.csv"
    if preferred.exists():
        invert_angular_z_csv(preferred, aug_dir / preferred.name)
    else:
        for c in orig_csvs:
            invert_angular_z_csv(c, aug_dir / c.name)

    print(f"[DONE] {aug_name} -> {aug_dir}")

# =========================
# Run
# =========================
if __name__ == "__main__":
    run_dark_versions([0.5, 0.75, 1.25])
    run_blur_versions(ksize=5, sigmas=[1.0])
    run_blur_versions(ksize=7, sigmas=[2.0, 3.0])
    run_contrast_versions(alphas=[0.60, 0.80])
    run_gamma_versions(gammas=[1.4])  
    # run_hflip()
