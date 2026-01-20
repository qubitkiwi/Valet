#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_dataset.py
---------------------
원본 dataset은 보존하고, out_dir에 전처리된 새 데이터셋 생성.

규칙:
1) -0.0 같은 값은 0.0으로 정규화
2) (linear_x==0.0 and angular_z==0.0) row는 삭제
   단, episode 마지막 keep_last 프레임은 (0,0)이어도 삭제하지 않음
3) 삭제된 row에 해당하는 front/rear 이미지는 out_dir로 복사하지 않음 (정합 유지)

추가(이번 요청):
4) 각 episode에서 "마지막 남은 프레임"의 front/rear 이미지를 stop_repeat 만큼 복제하고,
   actions.csv에 (0.0, 0.0) 라벨 row를 stop_repeat 만큼 추가한다.
   -> 주차 완료 후 정지 상태 학습용

출력 구조:
out_dir/
  run_000/
    episode_000/
      front/000001.jpg ...
      rear/000001.jpg ...
      actions.csv
  run_001_DAgger/...
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

HEADER = ["p1", "p2", "p3", "front_image", "rear_image", "linear_x", "angular_z"]

def _to_float(s: str) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return 0.0

def _norm_zero(x: float, eps: float) -> float:
    x = float(x)
    return 0.0 if abs(x) < eps else x

def _is_zero_pair(lx: float, az: float, eps: float) -> bool:
    return abs(lx) < eps and abs(az) < eps

def list_episode_dirs(dataset_dir: Path) -> List[Path]:
    runs = sorted([p for p in dataset_dir.glob("run_*") if p.is_dir()])
    eps: List[Path] = []
    for run in runs:
        for ep in sorted(run.glob("episode_*")):
            if not ep.is_dir():
                continue
            csv_path = ep / "actions.csv"
            front_dir = ep / "front"
            rear_dir = ep / "rear"
            if csv_path.exists() and front_dir.exists() and rear_dir.exists():
                eps.append(ep)
    return eps

def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if all(k in r for k in HEADER):
                rows.append(r)
    return rows

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))

def _parse_frame_id(path_str: str) -> Tuple[Optional[int], int]:
    """
    'front/000123.jpg' -> (123, 6)
    숫자 파싱 실패시 (None, 0)
    """
    p = Path(path_str)
    stem = p.stem  # '000123'
    if stem.isdigit():
        return int(stem), len(stem)
    return None, 0

def _format_frame_id(n: int, width: int) -> str:
    if width <= 0:
        return str(n)
    return str(n).zfill(width)

def _replace_filename_keep_dir(rel_path: str, new_stem: str) -> str:
    p = Path(rel_path)
    return str(p.parent / f"{new_stem}{p.suffix}")

def preprocess_episode_to_out(
    ep_in: Path,
    ep_out: Path,
    keep_last: int,
    zero_eps: float,
    strict_missing_images: bool,
    stop_repeat: int,
) -> Tuple[int, int, int]:
    """
    returns: (before_rows, after_rows, copied_images_count)
    """
    csv_in = ep_in / "actions.csv"
    rows = read_rows(csv_in)
    n = len(rows)
    if n == 0:
        return (0, 0, 0)

    cutoff = max(0, n - keep_last)

    # 출력 디렉토리 준비
    (ep_out / "front").mkdir(parents=True, exist_ok=True)
    (ep_out / "rear").mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, str]] = []
    copied = 0

    for i, r in enumerate(rows):
        p1 = _to_float(r["p1"])
        p2 = _to_float(r["p2"])
        p3 = _to_float(r["p3"])

        front_rel = str(r["front_image"]).strip()
        rear_rel  = str(r["rear_image"]).strip()

        lx = _norm_zero(_to_float(r["linear_x"]), eps=zero_eps)
        az = _norm_zero(_to_float(r["angular_z"]), eps=zero_eps)

        # (0,0) 필터: 마지막 keep_last 구간은 예외
        if _is_zero_pair(lx, az, eps=zero_eps) and (i < cutoff):
            continue

        # 이미지 존재 확인 + 복사
        front_src = (ep_in / front_rel).resolve()
        rear_src  = (ep_in / rear_rel).resolve()

        if not front_src.exists() or not rear_src.exists():
            if strict_missing_images:
                continue

        front_dst = (ep_out / front_rel).resolve()
        rear_dst  = (ep_out / rear_rel).resolve()

        safe_copy(front_src, front_dst)
        safe_copy(rear_src, rear_dst)
        copied += 2

        out_rows.append({
            "p1": f"{p1:.0f}",
            "p2": f"{p2:.0f}",
            "p3": f"{p3:.0f}",
            "front_image": front_rel,
            "rear_image": rear_rel,
            "linear_x": f"{lx:.6f}",
            "angular_z": f"{az:.6f}",
        })

    # ====== 추가: 정지 프레임 stop_repeat개 붙이기 ======
    if stop_repeat > 0 and len(out_rows) > 0:
        last = out_rows[-1]
        last_front_rel = last["front_image"]
        last_rear_rel  = last["rear_image"]

        # out_dir에 이미 복사된 "마지막 이미지"를 원본으로 사용 (정합/속도/안전)
        last_front_src = (ep_out / last_front_rel).resolve()
        last_rear_src  = (ep_out / last_rear_rel).resolve()

        if last_front_src.exists() and last_rear_src.exists():
            last_id, width = _parse_frame_id(last_front_rel)
            # 숫자 파싱 실패하면 그냥 1..N 형태로 뒤에 suffix를 붙여 생성
            if last_id is None:
                # 예: front/last.jpg -> front/last_stop_001.jpg
                for k in range(1, stop_repeat + 1):
                    new_stem_f = f"{Path(last_front_rel).stem}_stop_{str(k).zfill(3)}"
                    new_stem_r = f"{Path(last_rear_rel).stem}_stop_{str(k).zfill(3)}"

                    new_front_rel = _replace_filename_keep_dir(last_front_rel, new_stem_f)
                    new_rear_rel  = _replace_filename_keep_dir(last_rear_rel,  new_stem_r)

                    safe_copy(last_front_src, (ep_out / new_front_rel).resolve())
                    safe_copy(last_rear_src,  (ep_out / new_rear_rel).resolve())
                    copied += 2

                    out_rows.append({
                        "p1": last["p1"],
                        "p2": last["p2"],
                        "p3": last["p3"],
                        "front_image": new_front_rel,
                        "rear_image": new_rear_rel,
                        "linear_x": f"{0.0:.6f}",
                        "angular_z": f"{0.0:.6f}",
                    })
            else:
                start = last_id
                for k in range(1, stop_repeat + 1):
                    new_id = start + k
                    new_stem = _format_frame_id(new_id, width)

                    new_front_rel = _replace_filename_keep_dir(last_front_rel, new_stem)
                    new_rear_rel  = _replace_filename_keep_dir(last_rear_rel,  new_stem)

                    safe_copy(last_front_src, (ep_out / new_front_rel).resolve())
                    safe_copy(last_rear_src,  (ep_out / new_rear_rel).resolve())
                    copied += 2

                    out_rows.append({
                        "p1": last["p1"],
                        "p2": last["p2"],
                        "p3": last["p3"],
                        "front_image": new_front_rel,
                        "rear_image": new_rear_rel,
                        "linear_x": f"{0.0:.6f}",
                        "angular_z": f"{0.0:.6f}",
                    })

    # actions.csv 새로 작성
    csv_out = ep_out / "actions.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for rr in out_rows:
            w.writerow(rr)

    return (n, len(out_rows), copied)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default='/home/sechankim/ros2_ws/src/parking_server/dataset', help="원본 DATASET_DIR (run_*가 있는 폴더)")
    ap.add_argument("--out_dir", type=str, default='/home/sechankim/ros2_ws/src/parking_server/dataset/preprocessed_dataset', help="전처리 결과를 생성할 폴더")
    ap.add_argument("--keep_last", type=int, default=5, help="episode 마지막 N프레임은 (0,0)이어도 유지")
    ap.add_argument("--stop_repeat", type=int, default=5, help="각 episode 마지막 프레임을 복제해 (0,0) 라벨로 추가할 개수")
    ap.add_argument("--zero_eps", type=float, default=1e-6, help="0 판정 eps (abs(x)<eps => 0)")
    ap.add_argument("--overwrite", action="store_true", help="out_dir이 이미 있으면 삭제하고 새로 생성")

    # ✅ 증분 처리: 이미 처리한 episode는 스킵
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="(기본 True. 다시 만들 거면 overwrite를 cli로 줘야 함) out_dir에 actions.csv가 이미 있으면 해당 episode는 스킵(중복 전처리 방지)")

    ap.add_argument("--strict_missing_images", action="store_true", default=True,
                    help="(기본 True) front/rear 이미지 하나라도 없으면 해당 row 버림")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            # overwrite가 아니면 기존 out_dir을 유지하면서 증분 처리 가능
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    eps = list_episode_dirs(dataset_dir)
    if not eps:
        raise RuntimeError(f"no episodes under: {dataset_dir}/run_*/episode_*/(front,rear,actions.csv)")

    total_before = 0
    total_after = 0
    total_imgs = 0
    skipped = 0

    for ep_in in eps:
        run_in = ep_in.parent
        run_name = run_in.name
        ep_name = ep_in.name
        ep_out = out_dir / run_name / ep_name

        # ✅ 이미 처리된 episode는 스킵
        if args.skip_existing and (ep_out / "actions.csv").exists():
            skipped += 1
            continue

        before, after, copied = preprocess_episode_to_out(
            ep_in=ep_in,
            ep_out=ep_out,
            keep_last=args.keep_last,
            zero_eps=args.zero_eps,
            strict_missing_images=args.strict_missing_images,
            stop_repeat=args.stop_repeat,
        )

        total_before += before
        total_after += after
        total_imgs += copied

        print(f"[OK] {run_name}/{ep_name}: rows {before} -> {after}, copied_images={copied}")

    print(f"[DONE] rows: {total_before} -> {total_after} | copied_images={total_imgs} | skipped={skipped}")
    print(f"[DONE] out_dir: {out_dir}")

if __name__ == "__main__":
    main()
