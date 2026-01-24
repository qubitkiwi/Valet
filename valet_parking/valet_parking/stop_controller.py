#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# valet_parking/stop_session_controller.py

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


def next_run_index(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    mx = -1
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            # run_003_stop 같은 형태도 처리
            tail = p.name[4:]
            num_part = tail.split("_")[0]
            try:
                mx = max(mx, int(num_part))
            except Exception:
                pass
    return mx + 1


def next_episode_index(run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    mx = -1
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("episode_"):
            try:
                mx = max(mx, int(p.name.split("_")[1]))
            except Exception:
                pass
    return mx + 1


class StopSessionController(Node):
    """
    요구사항:
    - run: 노드 실행 시 1번 생성 (run_00x_stop)
    - episode: START 누를 때마다 증가 (episode_00y)
    - START: record_control=True + EP_START 발행 (0,0 포함 수집 시작)
    - END: record_control=False + EP_END 발행 (수집 종료)
    """

    def __init__(self):
        super().__init__('stop_session_controller')

        self.dataset_root = Path(str(self.declare_parameter(
            'dataset_root',
            str(Path('~/ros2_ws/src/dataset/valet_parking').expanduser())
        ).value)).expanduser()

        self.debounce_sec = float(self.declare_parameter('debounce_sec', 0.25).value)

        self.pub_evt  = self.create_publisher(String, '/valet/event', 10)
        self.pub_rec  = self.create_publisher(Bool,   '/record_control', 10)
        self.pub_mode = self.create_publisher(String, '/mux/select', 10)

        self.create_subscription(String, '/joy/event', self.on_button, 10)

        # ✅ run은 프로세스 시작 시 1번만 만든다
        idx = next_run_index(self.dataset_root)
        self.run_name = f'run_{idx:03d}_stop'
        self.run_dir = self.dataset_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.state = 'IDLE'  # IDLE / REC
        self.episode_name = None

        self._last_evt_t = {}

        # 기본값: 안전하게 수집 OFF
        self._set_record(False)
        self._set_mode('joy')

        self.get_logger().info(f'[STOP_CTRL] ready root={self.dataset_root} run={self.run_name}')

    def _publish_event(self, typ: str):
        payload = {
            "type": typ,
            "dagger": False,
            "run_name": self.run_name,
            "episode_name": self.episode_name,
        }
        self.pub_evt.publish(String(data=json.dumps(payload)))

    def _set_record(self, on: bool):
        self.pub_rec.publish(Bool(data=bool(on)))

    def _set_mode(self, m: str):
        # joy|policy|stop (cmd_mux_node가 받음)
        self.pub_mode.publish(String(data=m))

    def on_button(self, msg: String):
        try:
            evt = (json.loads(msg.data).get('type', '') or '').upper().strip()
        except Exception:
            self.get_logger().warn(f'[STOP_CTRL] bad /joy/event: {msg.data}')
            return

        # ✅ debounce
        now = time.monotonic()
        last = self._last_evt_t.get(evt, 0.0)
        if (now - last) < self.debounce_sec:
            return
        self._last_evt_t[evt] = now

        if evt == 'START':
            if self.state != 'IDLE':
                return

            ep_idx = next_episode_index(self.run_dir)
            self.episode_name = f'episode_{ep_idx:03d}'

            # 수집 시작
            self._set_mode('joy')        # 기록 중에도 조이스틱 모드 유지(원하면 stop으로 바꿔도 됨)
            self._set_record(True)
            self._publish_event('EP_START')
            self.state = 'REC'

            self.get_logger().info(f'[STOP_CTRL] EP_START run={self.run_name} ep={self.episode_name}')

        elif evt == 'END':
            if self.state != 'REC':
                return

            # 수집 종료 + 안전 정지
            self._set_record(False)
            self._set_mode('stop')
            self._publish_event('EP_END')
            self.state = 'IDLE'

            self.get_logger().info(f'[STOP_CTRL] EP_END run={self.run_name} ep={self.episode_name}')

        else:
            # X 같은건 무시
            return


def main(args=None):
    rclpy.init(args=args)
    node = StopSessionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
