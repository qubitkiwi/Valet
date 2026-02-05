#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

# ============================= 설정 ===============================
BASE_DIR = Path("/home/elicer/jun_ws") # 기본 작업 경로
CSV_FILENAME = "total_data_final_v2.csv"  # CSV 파일명 (모든 폴더에서 동일)

# 데이터 디렉토리 (BASE 기준 상대 경로)
DIR_CONFIG = {
    "original": "data/total_data_final",             # 원본 데이터 위치
    "augmented": "data_augmented/total_data_final",  # 증강 데이터 루트 위치
    "output": "model_train/total_data_final_v2"         # 병합 결과 저장 위치

}
# CSV 컬럼 헤더 (데이터 형식이 바뀌면 여기서 수정)
CSV_HEADERS = ["timestamp", "front_img", "rear_img", "left_img", "right_img", "linear_x", "angular_z", "sign_class"]

# 포함할 Augmentation 폴더 목록 (Suffix로도 사용됨)
AUGMENTATION_LIST = [
    "dark_1.250",
    "dark_0.750",
    "dark_0.500",
    "contrast_a0.800",
    "contrast_a0.600",
    "gamma_g1.400",
    "blur_k5_sig1.000",
    "blur_k7_sig2.000",
    "blur_k7_sig3.000",
]
# ================================================================

def add_suffix(filename: str, suffix: str) -> str:
    p = Path(filename)
    return f"{p.stem}_{suffix}{p.suffix}"


def read_csv_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        required = set(CSV_HEADERS)
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{csv_path} missing columns: {required}")
        return list(r)


def write_csv_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # 헤더 쓰기
        w.writerow(CSV_HEADERS)
        # 모든 행 쓰기
        for r in rows:
            row_data = [r[col] for col in CSV_HEADERS]
            w.writerow(row_data)


def copy_image(src: Path, dst: Path, dry_run: bool):
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():

    default_orig = BASE_DIR / DIR_CONFIG["original"]
    default_aug = BASE_DIR / DIR_CONFIG["augmented"]
    default_out = BASE_DIR / DIR_CONFIG["output"]

    p = argparse.ArgumentParser(
        description="Merge original + selected augmentations into model_train with suffix-renamed images."
    )
    p.add_argument("--orig-dir", default=str(default_orig), help=f"Default: {default_orig}")
    p.add_argument("--aug-root", default=str(default_aug), help=f"Default: {default_aug}")
    p.add_argument("--out-dir", default=str(default_out), help=f"Default: {default_out}")
    p.add_argument("--csv-name", default=CSV_FILENAME, help=f"Default: {CSV_FILENAME}")
    p.add_argument("--write", action="store_true", help="Actually copy files and write CSV (default: dry-run)")
    args = p.parse_args()

    dry_run = not args.write  # 기본값은 dry-run

    orig_dir = Path(args.orig_dir)
    aug_root = Path(args.aug_root)
    out_dir = Path(args.out_dir)
    out_img_dir = out_dir / "images"
    out_csv = out_dir / args.csv_name

    # Augmentations to include (suffix == aug folder name by default)
    aug_sets = AUGMENTATION_LIST

    merged_rows = []
    seen_filenames = set()

    # Original
    orig_csv = orig_dir / args.csv_name
    orig_rows = read_csv_rows(orig_csv)
    for r in orig_rows:
        base = Path(r["front_img"]).name
        out_name = base
        if out_name in seen_filenames:
            raise ValueError(f"Duplicate filename detected: {out_name}")
        seen_filenames.add(out_name)
        src = orig_dir / "images" / base
        dst = out_img_dir / out_name
        copy_image(src, dst, dry_run)

        new_row = r.copy()
        merged_rows.append(new_row)

    # Augmented
    for aug_name in aug_sets:
        aug_dir = aug_root / aug_name
        aug_csv = aug_dir / args.csv_name
        if not aug_csv.exists():
            raise FileNotFoundError(f"Missing CSV: {aug_csv}")

        aug_rows = read_csv_rows(aug_csv)
        for r in aug_rows:
            base = Path(r["front_img"]).name
            out_name = add_suffix(base, aug_name)
            if out_name in seen_filenames:
                raise ValueError(f"Duplicate filename detected: {out_name}")
            seen_filenames.add(out_name)
            src = aug_dir / "images" / base
            dst = out_img_dir / out_name
            copy_image(src, dst, dry_run)
            
            new_row = r.copy()
            new_row["front_img"] = f"images/{out_name}"
            merged_rows.append(new_row)

    merged_rows.sort(key=lambda r: r["front_img"])

    if dry_run:
        print(f"[DRY RUN] would write {len(merged_rows)} rows to {out_csv}")
        print(f"[DRY RUN] would copy images into {out_img_dir}")
    else:
        write_csv_rows(out_csv, merged_rows)
        print(f"[DONE] wrote {len(merged_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
