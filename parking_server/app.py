#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel

# inference.py의 infer() 사용
from inference import infer

app = FastAPI()


def now_str():
    return time.strftime("%H:%M:%S")


def log(msg: str):
    print(f"[{now_str()}] {msg}", flush=True)


# ✅ 너가 요구한 헤더 “그대로 고정”
CSV_HEADER = "p1,p2,p3,front_image,rear_image,linear_x,angular_z\n"


def slot_onehot_p1twice(slot_name: str):
    """
    헤더가 p1,p2,p3 이므로:
      - col1(p1)
      - col2(p2)
      - col3(p3)
    """
    s = (slot_name or "p0").lower().strip()
    p1 = 1 if s == "p1" else 0
    p2 = 1 if s == "p2" else 0
    p3 = 1 if s == "p3" else 0
    return p1, p2, p3


class Event(BaseModel):
    type: str
    slot: Optional[int] = None
    slot_name: Optional[str] = None
    dagger: Optional[bool] = None
    ts: Optional[float] = None
# AXES_MAP = ['lx', 'ly', 'rx', 'ry', 'r2', 'l2', 'hat_x', 'hat_y']
# BUTTON_MAP = ['Y', 'B', 'A', 'X', 'l1', 'r1', 'l2', 'r2', 'select', 'start', 'l3', 'r3','mode']

STATE = {
    # episode state
    "armed_episode": False,     # EP_START 받음 (폴더는 아직)
    "episode_created": False,   # REC_ON에서 폴더 생성했는지
    "recording": False,         # REC_ON~REC_OFF 동안만 True
    "dagger": False,
    # "recording_until": 0.0,

    # run/episode
    "base_dir": "/home/sechankim/ros2_ws/src/parking_server/dataset",  # 서버 머신 기준 경로!
    "run_name": None,
    "ep_idx": 0,
    "ep_dir": None,

    # slot
    "slot": 0,
    "slot_name": "p0",

    # stats
    "ep_saved_total": 0,
    "skipped_not_recording": 0,
    "seg_saved": 0,
    "seg_t0": None,

    # policy logging throttling
    "last_policy_log_t": 0.0,
    "policy_log_hz": 2.0,  # 초당 2회 정도만 찍기
}

def _ensure_base_dir():
    bd = STATE["base_dir"]
    if not os.path.exists(bd):
        # base_dir 자체가 없으면 생성 시도
        os.makedirs(bd, exist_ok=True)
    if not os.path.isdir(bd):
        raise RuntimeError(f"base_dir is not dir: {bd}")


def _choose_run_name(dagger: bool) -> str:
    """
    run_000 / run_000_DAgger 형태로 자동 증가.
    서버 base_dir 아래에 같은 suffix끼리만 카운트.
    """
    prefix = "run_"
    suffix = "_DAgger" if dagger else ""
    base = STATE["base_dir"]

    # existing: run_??? + suffix만
    existing = []
    for name in os.listdir(base):
        if not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        if (not suffix) and name.endswith("_DAgger"):
            continue
        existing.append(name)

    n = len(sorted(existing))
    return f"{prefix}{n:03d}{suffix}"


def ensure_run_and_episode_created():
    """
    ✅ 첫 REC_ON 때만 호출
    - run 생성(없으면)
    - episode 폴더 생성
    - actions.csv 생성(헤더 고정)
    """
    if STATE["episode_created"]:
        return

    _ensure_base_dir()

    dagger = bool(STATE["dagger"])

    # run_name 없으면 새 run 생성
    if STATE["run_name"] is None:
        STATE["run_name"] = _choose_run_name(dagger)
        run_dir = os.path.join(STATE["base_dir"], STATE["run_name"])
        os.makedirs(run_dir, exist_ok=False)
        log(f"[STATE] New run created: {STATE['run_name']} dagger={dagger}")

    run_dir = os.path.join(STATE["base_dir"], STATE["run_name"])
    ep_dir = os.path.join(run_dir, f"episode_{STATE['ep_idx']:03d}")

    os.makedirs(os.path.join(ep_dir, "front"), exist_ok=True)
    os.makedirs(os.path.join(ep_dir, "rear"), exist_ok=True)

    csv_path = os.path.join(ep_dir, "actions.csv")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)

    STATE["ep_dir"] = ep_dir
    STATE["episode_created"] = True

    log(f"[STATE] Episode created on first REC_ON: {ep_dir}")


def _seg_on():
    STATE["seg_saved"] = 0
    STATE["seg_t0"] = time.time()


def _seg_off():
    t0 = STATE["seg_t0"]
    dt = (time.time() - t0) if t0 else 0.0
    seg_saved = STATE["seg_saved"]
    total = STATE["ep_saved_total"]
    log(f"[REC] OFF (segment) dt={dt:.2f}s seg_saved={seg_saved} (ep_saved_total={total}) dagger={STATE['dagger']}")
    STATE["seg_t0"] = None
    STATE["seg_saved"] = 0


@app.post("/collector/event")
async def collector_event(req: Request):
    ev = await req.json()
    t = (ev.get("type") or "").upper().strip()

    # dagger flag
    if "dagger" in ev:
        STATE["dagger"] = bool(ev.get("dagger"))

    # SLOT 저장
    if t == "SLOT":
        STATE["slot"] = int(ev.get("slot", 0) or 0)
        STATE["slot_name"] = (ev.get("slot_name") or STATE["slot_name"] or "p0")
        log(f"[EP] SLOT selected: slot={STATE['slot']} name={STATE['slot_name']} dagger={STATE['dagger']}")
        return {"ok": True}

    if t == "EP_START":
        # ✅ 폴더 생성 X, armed만
        STATE["armed_episode"] = True
        STATE["episode_created"] = False
        STATE["recording"] = False
        STATE["ep_dir"] = None

        STATE["slot"] = 0
        STATE["slot_name"] = "p0"

        # stats reset (episode 단위)
        STATE["ep_saved_total"] = 0
        STATE["skipped_not_recording"] = 0
        STATE["seg_saved"] = 0
        STATE["seg_t0"] = None

        log(f"[EP] START armed (ep_idx={STATE['ep_idx']:03d}) dagger={STATE['dagger']} (no folder created)")
        return {"ok": True}

    if t == "REC_ON":
        if not STATE["armed_episode"]:
            STATE["armed_episode"] = True  # 방어적

        ensure_run_and_episode_created()
        STATE["recording"] = True
        _seg_on()

        # slot이 payload로 들어오면 slot_name 반영
        if ev.get("slot_name"):
            STATE["slot_name"] = ev["slot_name"]

        log(f"[REC] ON  (ep_idx={STATE['ep_idx']:03d}) slot={STATE['slot_name']} dagger={STATE['dagger']} -> recording=True")
        return {"ok": True}

    if t == "REC_OFF":
        STATE["recording"] = False
        _seg_off()
        return {"ok": True}

    if t == "EP_END":
        # recording 종료
        if STATE["recording"]:
            STATE["recording"] = False
            _seg_off()

        STATE["armed_episode"] = False

        if STATE["episode_created"]:
            # episode가 실제로 만들어졌으면 다음 ep로
            log(
                f"[EP] END ep_idx={STATE['ep_idx']:03d} dagger={STATE['dagger']} "
                f"saved={STATE['ep_saved_total']} skipped_not_recording={STATE['skipped_not_recording']} -> next episode"
            )
            STATE["ep_idx"] += 1
        else:
            log(f"[EP] END ep_idx={STATE['ep_idx']:03d} dagger={STATE['dagger']} (no intervention) -> nothing created")

        # reset for safety
        STATE["episode_created"] = False
        STATE["ep_dir"] = None
        STATE["recording"] = False
        STATE["slot"] = 0
        STATE["slot_name"] = "p0"

        return {"ok": True}

    # 기타 이벤트는 그냥 ok
    return {"ok": True}


@app.post("/collector/frame")
async def collector_frame(
    front: UploadFile = File(...),
    rear: UploadFile = File(...),
    slot: str = Form(...),         # 브릿지에서 보낸 값(참고용)
    frame_idx: str = Form(...),
    linear_x: str = Form(...),
    angular_z: str = Form(...),
):
    # ✅ recording 아닐 때는 저장하지 않음 (빈폴더도 안 생기게: 폴더 생성은 REC_ON에서만)
    if not STATE["recording"]:
        STATE["skipped_not_recording"] += 1
        return {"ok": True, "skipped": "not_recording"}

    # recording인데 episode 폴더가 없다? 방어적으로 생성
    if not STATE.get("episode_created", False):
        ensure_run_and_episode_created()

    ep_dir = STATE["ep_dir"]
    idx = int(frame_idx)

    front_name = f"{idx:06d}.jpg"
    rear_name = f"{idx:06d}.jpg"

    front_path = os.path.join(ep_dir, "front", front_name)
    rear_path = os.path.join(ep_dir, "rear", rear_name)

    fb = await front.read()
    rb = await rear.read()

    with open(front_path, "wb") as f:
        f.write(fb)
    with open(rear_path, "wb") as f:
        f.write(rb)

    # actions.csv append
    csv_path = os.path.join(ep_dir, "actions.csv")
    if (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)

    p1, p2, p3 = slot_onehot_p1twice(slot)
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(f"{p1},{p2},{p3},front/{front_name},rear/{rear_name},{linear_x},{angular_z}\n")

    STATE["ep_saved_total"] += 1
    STATE["seg_saved"] += 1

    return {"ok": True}


@app.post("/infer")
async def infer_api(
    front: UploadFile = File(...),
    rear: UploadFile = File(...),
    slot_name: str = Form("p0"),
):
    fb = await front.read()
    rb = await rear.read()

    v, w = infer(fb, rb, slot_name)

    # 너무 도배되면 throttle
    t = time.time()
    if t - STATE["last_policy_log_t"] > (1.0 / max(0.1, STATE["policy_log_hz"])):
        STATE["last_policy_log_t"] = t
        log(f"[POLICY] slot={slot_name} v={v:+.3f} w={w:+.3f}")

    return {"linear_x": float(v), "angular_z": float(w)}
