#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
joystick_event_publisher.py
- /dev/input/js0 에서 joystick 이벤트를 읽어 /cmd_vel_joy publish
- 조이스틱 "활성(내가 조작 중)" 여부를 /joy/active(Bool)로 publish
- 핵심: JOY_ACTIVE 튐 방지
  1) active 판정은 0.0이 아니라 threshold 기반
  2) cmd/active publish는 타이머(20Hz)에서만 수행 (이벤트 유무와 무관하게 일정)
  3) axis 이벤트는 상태 업데이트만 수행
"""

import os
import struct
import threading
import time
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
# AXES_MAP = ['lx', 'ly', 'rx', 'ry', 'r2', 'l2', 'hat_x', 'hat_y']
# BUTTON_MAP = ['Y', 'B', 'A', 'X', 'l1', 'r1', 'l2', 'r2', 'select', 'start', 'l3', 'r3','mode']
# ==========================
# 버튼 매핑 (필요하면 숫자만 바꾸면 됨)
# 확인: jstest /dev/input/js0
# ==========================
BTN_EP_START = 9    # start
BTN_EP_END   = 8    # select
BTN_SLOT_1   = 3    # X
BTN_SLOT_2   = 0    # Y
BTN_SLOT_3   = 1    # B


def map_range(x, in_min, in_max, out_min, out_max):
    # in_min == in_max 방지
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class JoystickEventPublisher(Node):
    def __init__(self):
        super().__init__('joystick_event_publisher')

        # axis mapping
        self.declare_parameter('axis_linear', 1)
        self.declare_parameter('axis_angular', 2)
        self.AX_LY = int(self.get_parameter('axis_linear').value)
        self.AX_RX = int(self.get_parameter('axis_angular').value)

        self.declare_parameter('invert_angular', False)   # ✅ 기본 True 추천
        self.declare_parameter('invert_linear',  False)  # 필요하면
        self.invert_ang = bool(self.get_parameter('invert_angular').value)
        self.invert_lin = bool(self.get_parameter('invert_linear').value)

        # ranges & device
        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 3.0)
        self.declare_parameter('js_dev', '/dev/input/js0')

        # deadzone / active logic
        self.declare_parameter('deadzone', 0.10)
        self.declare_parameter('active_hold_sec', 0.80)
        self.declare_parameter('active_threshold', 0.15)  # deadzone보다 약간 크게 추천

        # publish rate
        self.declare_parameter('publish_hz', 20.0)

        self.max_linear = float(self.get_parameter('max_linear').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.js_dev = str(self.get_parameter('js_dev').value)

        self.deadzone = float(self.get_parameter('deadzone').value)
        self.active_hold_sec = float(self.get_parameter('active_hold_sec').value)
        self.active_threshold = float(self.get_parameter('active_threshold').value)

        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.dt = 1.0 / max(self.publish_hz, 1.0)

        # pubs
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_joy', 10)
        self.pub_evt_raw = self.create_publisher(String, '/parking/event_raw', 10)
        self.pub_active = self.create_publisher(Bool, '/joy/active', 10)

        # state
        self.axes = [0.0] * 8
        self.buttons = [0] * 16

        self.last_stick_active_wall = 0.0
        self.active_state = False

        # timer: cmd/active publish는 여기서만 한다 (중요)
        self.create_timer(self.dt, self._tick)

        # joystick reader thread
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

        self.get_logger().info(
            f'JoystickEventPublisher started: {self.js_dev} | '
            f'publish_hz={self.publish_hz} deadzone={self.deadzone} '
            f'active_threshold={self.active_threshold} active_hold_sec={self.active_hold_sec} '
            f'axis_linear={self.AX_LY} axis_angular={self.AX_RX}'
        )

    # ---------- raw event publishers ----------
    def _emit_raw_obj(self, obj: dict):
        m = String()
        m.data = json.dumps(obj, ensure_ascii=False)
        self.pub_evt_raw.publish(m)

    def _emit_raw_str(self, s: str):
        m = String()
        m.data = s
        self.pub_evt_raw.publish(m)

    # ---------- button handler ----------
    def _on_button(self, idx: int):
        self.get_logger().info(f'[BUTTON] pressed idx={idx}')

        if idx == BTN_EP_START:
            self.get_logger().info('[EP] START pressed')
            self._emit_raw_obj({"type": "BTN", "name": "EP_START"})
            self._emit_raw_str('{"type":"EP_START"}')

        elif idx == BTN_EP_END:
            self.get_logger().info('[EP] END pressed')
            self._emit_raw_obj({"type": "BTN", "name": "EP_END"})
            self._emit_raw_str('{"type":"EP_END"}')

        elif idx == BTN_SLOT_1:
            self.get_logger().info('[SLOT] P1 pressed')
            self._emit_raw_obj({"type": "BTN", "name": "P1"})
            self._emit_raw_str('{"type":"SLOT","slot":1}')

        elif idx == BTN_SLOT_2:
            self.get_logger().info('[SLOT] P2 pressed')
            self._emit_raw_obj({"type": "BTN", "name": "P2"})
            self._emit_raw_str('{"type":"SLOT","slot":2}')

        elif idx == BTN_SLOT_3:
            self.get_logger().info('[SLOT] P3 pressed')
            self._emit_raw_obj({"type": "BTN", "name": "P3"})
            self._emit_raw_str('{"type":"SLOT","slot":3}')

    # ---------- stick active 판단 (threshold 기반) ----------
    def _is_stick_active(self) -> bool:
        ly = -self.axes[self.AX_LY]
        rx = self.axes[self.AX_RX]
        th = self.active_threshold
        return (abs(ly) >= th) or (abs(rx) >= th)

    # ---------- cmd publish ----------
    def _publish_cmd(self):
        lin = map_range(-self.axes[self.AX_LY], -1, 1, -self.max_linear, self.max_linear)
        ang = map_range(self.axes[self.AX_RX], -1, 1, -self.max_angular, self.max_angular)

        if self.invert_lin:
            lin = -lin
        if self.invert_ang:
            ang = -ang


        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.pub_cmd.publish(t)

    # ---------- active publish ----------
    def _publish_active(self, active: bool):
        if active != self.active_state:
            self.active_state = active
            self.get_logger().info(f"[JOY_ACTIVE] {active}")

        b = Bool()
        b.data = bool(active)
        self.pub_active.publish(b)

    # ---------- timer tick (20Hz) ----------
    def _tick(self):
        # 1) 항상 일정 주기로 cmd publish
        self._publish_cmd()

        # 2) 현재 axes 기반으로 active 갱신
        now = time.time()
        if self._is_stick_active():
            self.last_stick_active_wall = now

        active = (now - self.last_stick_active_wall) < self.active_hold_sec
        self._publish_active(active)

    # ---------- joystick reader thread ----------
    def _loop(self):
        if not os.path.exists(self.js_dev):
            self.get_logger().error(f'Joystick not found: {self.js_dev}')
            return

        try:
            with open(self.js_dev, 'rb') as f:
                while self.running and rclpy.ok():
                    ev = f.read(8)
                    if not ev:
                        break

                    _, value, type_, number = struct.unpack('IhBB', ev)
                    real = type_ & ~0x80

                    if real == 1:  # button
                        if number < len(self.buttons):
                            pressed = 1 if value else 0
                            self.buttons[number] = pressed
                            if pressed:
                                self._on_button(number)

                    elif real == 2:  # axis
                        if number < len(self.axes):
                            v = value / 32767.0
                            if abs(v) < self.deadzone:
                                v = 0.0
                            self.axes[number] = v
                            # 여기서는 publish 하지 않는다 (중요: publish는 타이머에서만)

        except PermissionError:
            self.get_logger().error(f'Permission denied: {self.js_dev} (sudo 필요할 수 있음)')
        except Exception as e:
            self.get_logger().error(f'Joystick read error: {e}')

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JoystickEventPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == '__main__':
    main()
