#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import threading
import requests
from collections import deque
from typing import Optional, Deque, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String


def slot_to_str(slot: int) -> str:
    if slot == 1:
        return "p1"
    if slot == 2:
        return "p2"
    if slot == 3:
        return "p3"
    return "p0"


def stamp_to_sec(stamp) -> float:
    # builtin_interfaces/msg/Time
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class CollectorHttpBridge(Node):
    def __init__(self):
        super().__init__('collector_http_bridge')

        self.declare_parameter('collector_event_url', 'http://192.168.0.10:8000/collector/event')
        self.declare_parameter('collector_frame_url', 'http://192.168.0.10:8000/collector/frame')
        self.declare_parameter('send_hz', 10.0)

        self.declare_parameter('dagger', False)
        self.declare_parameter('record_full_episode_when_not_dagger', True)

        self.declare_parameter('http_timeout_sec', 0.5)

        # REC_ON 직후 홀드오프 (이건 이벤트 기준이라 wall time 써도 OK)
        self.declare_parameter('recording_delay_sec', 0.2)

        # front/rear stamp 차이 허용
        self.declare_parameter('max_cam_dt_sec', 0.15)
        self.declare_parameter('require_synced_cams', True)

        # frame stamp 대비 label cmd 허용 오차 (너무 멀면 스킵)
        self.declare_parameter('max_cmd_dt_sec', 0.30)

        self.event_url = str(self.get_parameter('collector_event_url').value)
        self.frame_url = str(self.get_parameter('collector_frame_url').value)
        self.hz = float(self.get_parameter('send_hz').value)

        self.dagger = bool(self.get_parameter('dagger').value)
        self.record_full_episode_when_not_dagger = bool(self.get_parameter('record_full_episode_when_not_dagger').value)

        self.http_timeout = float(self.get_parameter('http_timeout_sec').value)
        self.recording_delay_sec = float(self.get_parameter('recording_delay_sec').value)

        self.max_cam_dt_sec = float(self.get_parameter('max_cam_dt_sec').value)
        self.require_synced_cams = bool(self.get_parameter('require_synced_cams').value)

        self.max_cmd_dt_sec = float(self.get_parameter('max_cmd_dt_sec').value)

        # ✅ mux_controller_dagger가 내는 최종 이벤트 토픽
        self.sub_evt = self.create_subscription(String, '/parking/event', self._evt, 10)

        # ✅ 카메라
        self.sub_f = self.create_subscription(CompressedImage, '/front/image/compressed', self._f, 1)
        self.sub_r = self.create_subscription(CompressedImage, '/rear/image/compressed',  self._r, 1)

        # ✅ 학습용 라벨 토픽 (cmd_mux가 publish): /mux/label_cmd (TwistStamped)
        self.sub_label = self.create_subscription(TwistStamped, '/mux/label_cmd', self._label_cmd, 50)

        # ---- state ----
        self.episode_active = False
        self.slot = 0
        self.slot_name = "p0"
        self.recording = False
        self.frame_idx = 0

        self.front = None
        self.rear = None
        self.front_t: Optional[float] = None   # ROS stamp sec
        self.rear_t: Optional[float] = None    # ROS stamp sec

        # REC_ON 이후 잠깐 저장 홀드오프(벽시계)
        self.recording_ready_at: float = 0.0

        # label buffer: (t_sec, v, steer_input)
        self.cmd_buf: Deque[Tuple[float, float, float]] = deque(maxlen=800)
        self.last_v = 0.0
        self.last_w = 0.0

        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

        self.get_logger().info(
            f'[READY] CollectorHttpBridge started | dagger={self.dagger} | '
            f'event_url={self.event_url} frame_url={self.frame_url} hz={self.hz} '
            f'max_cmd_dt_sec={self.max_cmd_dt_sec}'
        )

    # ------------- HTTP -------------
    def _post_event(self, obj: dict):
        try:
            requests.post(self.event_url, json=obj, timeout=self.http_timeout)
        except Exception:
            pass

    def _post_frame(self, files: dict, data: dict) -> bool:
        try:
            r = requests.post(self.frame_url, files=files, data=data, timeout=self.http_timeout)
            return (r.status_code == 200)
        except Exception:
            return False

    # ------------- event -------------
    def _evt(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except Exception:
            return

        t = str(obj.get('type', '')).upper().strip()

        if t == 'EP_START':
            self.episode_active = True
            self.slot = 0
            self.slot_name = "p0"
            self.frame_idx = 0
            self.recording = False

        elif t == 'EP_END':
            self.episode_active = False
            self.recording = False
            self.slot = 0
            self.slot_name = "p0"

        elif t == 'SLOT':
            self.slot = int(obj.get('slot', 0))
            self.slot_name = str(obj.get('slot_name') or slot_to_str(self.slot))

            if (not self.dagger) and self.record_full_episode_when_not_dagger and self.slot in (1, 2, 3):
                self.recording = True

        elif t == 'REC_ON':
            if self.dagger and self.slot in (1, 2, 3):
                self.recording = True
                self.recording_ready_at = time.time() + self.recording_delay_sec

        elif t == 'REC_OFF':
            if self.dagger:
                self.recording = False
                self.recording_ready_at = 0.0

        # ✅ 서버로 이벤트 전달(그대로)
        self._post_event(obj)

    # ------------- sensors -------------
    def _f(self, msg: CompressedImage):
        self.front = msg.data
        try:
            self.front_t = stamp_to_sec(msg.header.stamp)
        except Exception:
            self.front_t = None

    def _r(self, msg: CompressedImage):
        self.rear = msg.data
        try:
            self.rear_t = stamp_to_sec(msg.header.stamp)
        except Exception:
            self.rear_t = None

    def _label_cmd(self, msg: TwistStamped):
        # cmd_mux에서 now() 찍어서 보내는 stamp
        t = stamp_to_sec(msg.header.stamp)
        v = float(msg.twist.linear.x)
        w = float(msg.twist.angular.z)
        self.cmd_buf.append((t, v, w))
        self.last_v = v
        self.last_w = w

    # ------------- main loop -------------
    def _loop(self):
        dt = 1.0 / max(self.hz, 1.0)
        self.get_logger().info("Collector Loop Started")
        while self.running and rclpy.ok():
            time.sleep(dt)

            if not self.episode_active:
                continue
            if self.slot not in (1, 2, 3):
                self.get_logger().warn(f"Waiting for Slot (current: {self.slot})", once=True)
                continue
            if not self.recording:
                continue
            if self.recording_ready_at and (time.time() < self.recording_ready_at):
                continue
            if self.front is None or self.rear is None:
                self.get_logger().error("Recording is ON but Camera images are missing!")
                continue

            # ✅ 카메라 동기 체크 (ROS stamp 기반)
            if self.require_synced_cams:
                if (self.front_t is None) or (self.rear_t is None):
                    continue
                if abs(self.front_t - self.rear_t) > self.max_cam_dt_sec:
                    continue

            # ✅ frame 기준 시각(Front/Rear 평균)
            if (self.front_t is None) or (self.rear_t is None):
                continue
            ref_t = 0.5 * (self.front_t + self.rear_t)

            # ✅ 가장 가까운 label 선택
            v = self.last_v
            w = self.last_w
            if self.cmd_buf:
                t_near, v_near, w_near = min(self.cmd_buf, key=lambda x: abs(x[0] - ref_t))
                if abs(t_near - ref_t) > self.max_cmd_dt_sec:
                    # 프레임과 라벨이 너무 멀면 스킵 (정렬 노이즈 방지)
                    continue
                v, w = v_near, w_near
            else:
                # 아직 cmd가 없으면 스킵 (라벨 없는 샘플 방지)
                continue

            slot_send = self.slot_name if self.slot_name in ("p1", "p2", "p3") else slot_to_str(self.slot)

            files = {
                'front': ('front.jpg', bytes(self.front), 'image/jpeg'),
                'rear':  ('rear.jpg',  bytes(self.rear),  'image/jpeg'),
            }

            # ✅ 너가 원하는 필드만 전송/저장
            data = {
                'slot': slot_send,
                'frame_idx': str(self.frame_idx),
                'linear_x': str(float(v)),
                'angular_z': str(float(w)),
            }

            ok = self._post_frame(files=files, data=data)
            if ok:
                self.frame_idx += 1

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CollectorHttpBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
