#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


NEED_COLS = {"p1", "p2", "p3", "front_image", "rear_image", "linear_x", "angular_z"}


def list_episode_dirs(dataset_dir: Path) -> List[Path]:
    """
    지원하는 구조:
    1) (기존) run_000/episode_000/...
    2) (증강) run_000/orig/episode_000/...
            run_000/bri_0.5/episode_000/...
    """
    runs = sorted([p for p in dataset_dir.glob("run_*") if p.is_dir()])
    eps: List[Path] = []

    for run in runs:
        # case 1) run/episode_*
        for ep in sorted(run.glob("episode_*")):
            if ep.is_dir() and (ep / "actions.csv").exists() and (ep / "front").exists() and (ep / "rear").exists():
                eps.append(ep)

        # case 2) run/*/episode_*
        for ep in sorted(run.glob("*/episode_*")):
            if ep.is_dir() and (ep / "actions.csv").exists() and (ep / "front").exists() and (ep / "rear").exists():
                eps.append(ep)

    # 중복 제거(혹시라도 겹치면)
    eps = sorted(list(dict.fromkeys(eps)))
    return eps


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if r and NEED_COLS.issubset(set(r.keys())):
                rows.append(r)
        return rows


def episode_stats(ep_dir: Path) -> Tuple[int, int, int]:
    """
    returns:
      - total_rows: actions.csv row 수
      - valid_pairs: front/rear 둘 다 존재하는 row 수 (실제 학습에 쓰일 수 있는 수)
      - missing_pairs: row는 있는데 이미지 쌍이 깨진 수
    """
    csv_path = ep_dir / "actions.csv"
    rows = read_rows(csv_path)

    total_rows = len(rows)
    valid_pairs = 0

    for r in rows:
        try:
            front_rel = r["front_image"].strip()
            rear_rel = r["rear_image"].strip()
            fp = (ep_dir / front_rel).resolve()
            rp = (ep_dir / rear_rel).resolve()
            if fp.exists() and rp.exists():
                valid_pairs += 1
        except Exception:
            pass

    missing_pairs = total_rows - valid_pairs
    return total_rows, valid_pairs, missing_pairs


def _split_run_variant_episode(ep: Path) -> Tuple[str, str, str]:
    """
    ep 경로에서 (run_name, variant_name, episode_name) 뽑기
    - run_000/episode_000 -> (run_000, "", episode_000)
    - run_000/orig/episode_000 -> (run_000, "orig", episode_000)
    """
    episode_name = ep.name

    parent = ep.parent  # run_000 또는 variant
    if parent.name.startswith("run_"):
        # run_000/episode_000
        return parent.name, "", episode_name

    # run_000/<variant>/episode_000
    variant_name = parent.name
    run_dir = parent.parent
    run_name = run_dir.name if run_dir.name.startswith("run_") else "UNKNOWN_RUN"
    return run_name, variant_name, episode_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True, help="run_* 가 있는 폴더(예: aug_dataset)")
    ap.add_argument("--out_prefix", type=str, default="dataset_summary", help="출력 파일 prefix")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    episodes = list_episode_dirs(dataset_dir)
    if not episodes:
        raise RuntimeError(f"No episodes found under: {dataset_dir}/run_*/(episode_* or */episode_*)/actions.csv")

    out_csv = dataset_dir / f"{args.out_prefix}.csv"
    out_txt = dataset_dir / f"{args.out_prefix}.txt"

    # 집계
    run_totals: Dict[str, Dict[str, int]] = {}  # per run: eps, rows, valid, missing
    all_rows = all_valid = all_missing = 0

    rows_out = []
    for ep in episodes:
        run_name, variant_name, ep_name = _split_run_variant_episode(ep)

        total_rows, valid_pairs, missing_pairs = episode_stats(ep)

        all_rows += total_rows
        all_valid += valid_pairs
        all_missing += missing_pairs

        if run_name not in run_totals:
            run_totals[run_name] = {"episodes": 0, "rows": 0, "valid": 0, "missing": 0}
        run_totals[run_name]["episodes"] += 1
        run_totals[run_name]["rows"] += total_rows
        run_totals[run_name]["valid"] += valid_pairs
        run_totals[run_name]["missing"] += missing_pairs

        rows_out.append([
            run_name,
            variant_name if variant_name else "flat",
            ep_name,
            total_rows,
            valid_pairs,
            missing_pairs,
            str(ep),
        ])

    # episode 단위 CSV 저장
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "variant", "episode", "csv_rows", "valid_pairs", "missing_pairs", "episode_path"])
        # 유효쌍 많은 순으로 정렬해서 보기 좋게
        for r in sorted(rows_out, key=lambda x: x[4], reverse=True):
            w.writerow(r)

    # 한 파일(txt)로 “요약 + run별 + 상위/하위 episode”
    def pct(a, b):
        return (100.0 * a / b) if b > 0 else 0.0

    top5 = sorted(rows_out, key=lambda x: x[4], reverse=True)[:5]
    bot5 = sorted(rows_out, key=lambda x: x[4])[:5]

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"DATASET_DIR: {dataset_dir}\n")
        f.write(f"EPISODES: {len(episodes)}\n")
        f.write(f"TOTAL CSV ROWS: {all_rows}\n")
        f.write(f"TOTAL VALID PAIRS (train usable): {all_valid}\n")
        f.write(f"TOTAL MISSING PAIRS: {all_missing}\n")
        f.write(f"VALID RATE: {pct(all_valid, all_rows):.2f}%\n")
        f.write("\n")

        f.write("[RUN SUMMARY]\n")
        for run_name in sorted(run_totals.keys()):
            t = run_totals[run_name]
            f.write(
                f"- {run_name}: episodes={t['episodes']} rows={t['rows']} "
                f"valid={t['valid']} missing={t['missing']} valid_rate={pct(t['valid'], t['rows']):.2f}%\n"
            )

        f.write("\n[TOP 5 EPISODES BY VALID PAIRS]\n")
        for run, variant, ep, rows, valid, miss, path in top5:
            f.write(f"- {run}/{variant}/{ep}: valid={valid} rows={rows} missing={miss}\n  {path}\n")

        f.write("\n[BOTTOM 5 EPISODES BY VALID PAIRS]\n")
        for run, variant, ep, rows, valid, miss, path in bot5:
            f.write(f"- {run}/{variant}/{ep}: valid={valid} rows={rows} missing={miss}\n  {path}\n")

        f.write("\n(Details CSV) " + str(out_csv) + "\n")

    print(f"[OK] wrote:\n- {out_txt}\n- {out_csv}")


if __name__ == "__main__":
    main()
