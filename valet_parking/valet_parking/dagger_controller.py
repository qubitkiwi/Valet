#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from ros_robot_controller_msgs.msg import BuzzerState
from geometry_msgs.msg import Twist

def next_run_index(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    mx = -1
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
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


class DaggerController(Node):
    def __init__(self):
        super().__init__('dagger_controller')

        self.dagger = bool(self.declare_parameter('dagger', False).value)
        self.dataset_root = Path(str(self.declare_parameter(
            'dataset_root',
            str(Path('~/ros2_ws/src/dataset/valet_parking').expanduser())
        ).value)).expanduser()
        self.back_to_policy_v_th = float(self.declare_parameter('back_to_policy_v_th', 0.02).value)
        self.back_to_policy_w_th = float(self.declare_parameter('back_to_policy_w_th', 0.05).value)
        self.last_ctrl_cmd = (0.0, 0.0)
        self.episode_name = str(self.declare_parameter('episode_name', 'episode_000').value)

        self.beep_freq     = int(self.declare_parameter('beep_freq', 2500).value)
        self.beep_on_time  = float(self.declare_parameter('beep_on_time', 0.05).value)
        self.beep_off_time = float(self.declare_parameter('beep_off_time', 0.01).value)
        self.beep_repeat   = int(self.declare_parameter('beep_repeat', 1).value)

        # ✅ debounce 파라미터/상태
        self.debounce_sec = float(self.declare_parameter('debounce_sec', 0.25).value)
        self._last_evt_t = {}

        self.pub_mode = self.create_publisher(String, '/mux/select', 10)
        self.pub_rec = self.create_publisher(Bool, '/record_control', 10)
        self.pub_evt = self.create_publisher(String, '/valet/event', 10)
        self.pub_buzzer = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 10)

        self.create_subscription(String, '/joy/event', self.on_button, 10)
        self.create_subscription(Twist, '/controller/cmd_vel', self.on_ctrl_cmd, 10)  
        self.state = 'IDLE'
        self.run_name = None

        self.get_logger().info(f'[DAGGER] ready dagger={self.dagger} root={self.dataset_root}')

    def on_ctrl_cmd(self, msg: Twist):
        self.last_ctrl_cmd = (float(msg.linear.x), float(msg.angular.z))

    def _is_stopped(self) -> bool:
        v, w = self.last_ctrl_cmd
        return (abs(v) <= self.back_to_policy_v_th) and (abs(w) <= self.back_to_policy_w_th)

    def _beep(self, kind: str):
        freq_map = {'START': 2600, 'X': 1800, 'END': 1200}
        msg = BuzzerState()
        msg.freq = int(freq_map.get(kind, self.beep_freq))
        msg.on_time = float(self.beep_on_time)
        msg.off_time = float(self.beep_off_time)
        msg.repeat = int(self.beep_repeat)
        self.pub_buzzer.publish(msg)

    def _set_mode(self, m: str):
        self.pub_mode.publish(String(data=m))

    def _set_record(self, on: bool):
        self.pub_rec.publish(Bool(data=bool(on)))

    def _publish_valet_event(self, typ: str):
        payload = {
            "type": typ,
            "dagger": bool(self.dagger),
            "run_name": self.run_name,
            "episode_name": self.episode_name,
        }
        self.pub_evt.publish(String(data=json.dumps(payload)))

    def on_button(self, msg: String):
        try:
            evt = (json.loads(msg.data).get('type', '') or '').upper().strip()
        except Exception:
            self.get_logger().warn(f'[DAGGER] bad /joy/event: {msg.data}')
            return

        # ✅ debounce: 같은 이벤트가 짧은 시간에 연속 들어오면 무시
        now = time.monotonic()
        last = self._last_evt_t.get(evt, 0.0)
        if (now - last) < self.debounce_sec:
            return
        self._last_evt_t[evt] = now

        if evt == 'START':
            if self.state != 'IDLE':
                return

            # ✅ run은 세션 동안 고정 (없을 때만 생성)
            if self.run_name is None:
                idx = next_run_index(self.dataset_root)
                suffix = '_DAgger' if self.dagger else ''
                self.run_name = f'run_{idx:03d}{suffix}'

            run_dir = self.dataset_root / self.run_name

            # ✅ episode만 증가
            ep_idx = next_episode_index(run_dir)
            self.episode_name = f'episode_{ep_idx:03d}'

            self._publish_valet_event('EP_START')

            if self.dagger:
                self._set_record(False)
                self._set_mode('policy')
                self.state = 'EP_POLICY'
            else:
                self._set_record(True)
                self._set_mode('joy')
                self.state = 'EP_JOY'

            self._beep('START')
            self.get_logger().info(f'[DAGGER] EP_START run={self.run_name} ep={self.episode_name} state={self.state}')

        elif evt == 'X':
            if not self.dagger or self.state == 'IDLE':
                return

            if self.state == 'EP_POLICY':
                # policy -> joy (개입)
                self._set_mode('joy')
                self._set_record(True)
                self.state = 'EP_JOY'
                self._beep('X')
            else:
                # joy -> policy (회복)
                if not self._is_stopped():
                    # 아직 조이스틱 움직이는 중이면 무시
                    self.get_logger().warn('[DAGGER] cannot back to policy: joystick still moving')
                    return

                self._set_record(False)
                self._set_mode('policy')
                self.state = 'EP_POLICY'
                self._beep('X')

        elif evt == 'END':
            if self.state == 'IDLE':
                return

            self._set_record(False)
            self._set_mode('stop')
            self._publish_valet_event('EP_END')
            self._beep('END')

            self.get_logger().info(f'[DAGGER] EP_END run={self.run_name} ep={self.episode_name}')
            self.state = 'IDLE'
            # ✅ run_name 유지 (원하면 여기서 None 처리)


def main(args=None):
    rclpy.init(args=args)
    node = DaggerController()
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
