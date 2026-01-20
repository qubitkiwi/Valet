#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, time, threading, requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from ros_robot_controller_msgs.msg import BuzzerState


class PolicyHttpClient(Node):
    def __init__(self):
        super().__init__('policy_http_client')

        self.declare_parameter('server_url', 'http://192.168.0.10:8000')
        self.declare_parameter('rate_hz', 10.0)

        # ✅ 조향/전진 부호 반전 옵션 (한 곳에서만 적용!)
        self.declare_parameter('invert_angular', False)   # ✅ 조향 반대면 True
        self.declare_parameter('invert_linear',  False)  # 필요하면 True

        base = self.get_parameter('server_url').value.rstrip('/')
        self.infer_url = f"{base}/infer"
        self.rate = float(self.get_parameter('rate_hz').value)

        self.invert_w = bool(self.get_parameter('invert_angular').value)
        self.invert_v = bool(self.get_parameter('invert_linear').value)

        # --- subs ---
        self.sub_f = self.create_subscription(CompressedImage, '/front/image/compressed', self._f, 1)
        self.sub_r = self.create_subscription(CompressedImage, '/rear/image/compressed',  self._r, 1)

        # ✅ mux_controller가 내는 이벤트 토픽
        self.sub_evt = self.create_subscription(String, '/parking/event', self._evt, 10)

        # ✅ 제일 중요: cmd_mux 선택 토픽은 /cmd_mux/select
        self.sub_mux = self.create_subscription(String, '/mux/select', self._mux, 10)

        # --- pubs ---
        self.pub = self.create_publisher(Twist, '/cmd_vel_policy', 10)
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 10)

        # --- state ---
        self.front = None
        self.rear  = None
        self.slot = "p0"
        self.episode_active = False
        self.mux_mode = 'stop'  # 초기값 stop이 안전

        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

        self.get_logger().info(
            f"[READY] PolicyHttpClient infer_url={self.infer_url} rate={self.rate} "
            f"invert_v={self.invert_v} invert_w={self.invert_w}"
        )

    def _f(self, msg): self.front = msg.data
    def _r(self, msg): self.rear  = msg.data
    def _mux(self, msg): self.mux_mode = (msg.data or "").strip()

    def _beep(self, freq=2000, repeat=1):
        try:
            msg = BuzzerState()
            msg.freq = int(freq)
            msg.on_time = 0.05
            msg.off_time = 0.05
            msg.repeat = int(repeat)
            self.buzzer_pub.publish(msg)
        except Exception:
            pass

    def _evt(self, msg):
        try:
            obj = json.loads(msg.data)
        except Exception:
            return

        t = (obj.get('type') or '').strip()

        if t == 'EP_START':
            self.episode_active = True
            self.slot = "p0"
            self._beep(freq=2500, repeat=1)

        elif t == 'EP_END':
            self.episode_active = False
            self.slot = "p0"
            self._beep(freq=1200, repeat=1)

        elif t == 'SLOT':
            # ✅ mux가 보내는 형태: slot(int) + slot_name("p2") 둘 다 올 수 있음
            sn = (obj.get('slot_name') or '').lower().strip()
            if sn in ('p1', 'p2', 'p3'):
                self.slot = sn
            else:
                s = obj.get('slot', 'p0')
                # int면 p{int}로, str이면 처리
                try:
                    si = int(s)
                    self.slot = f"p{si}" if si in (1,2,3) else "p0"
                except Exception:
                    ss = str(s).lower().strip()
                    if ss in ('p1','p2','p3'):
                        self.slot = ss
                    elif ss in ('1','2','3'):
                        self.slot = f"p{ss}"
                    else:
                        self.slot = "p0"

            self._beep(freq=1700, repeat=1)

    def _loop(self):
        dt = 1.0 / max(self.rate, 1.0)
        while self.running and rclpy.ok():
            time.sleep(dt)

            # ✅ mux가 policy일 때만 publish
            if self.mux_mode != 'policy':
                continue
            if not self.episode_active:
                continue
            if self.slot not in ('p1','p2','p3'):
                continue
            if self.front is None or self.rear is None:
                continue

            try:
                files = {
                    'front': ('front.jpg', bytes(self.front), 'image/jpeg'),
                    'rear':  ('rear.jpg',  bytes(self.rear),  'image/jpeg'),
                }
                r = requests.post(
                    self.infer_url,
                    files=files,
                    data={'slot_name': self.slot},
                    timeout=0.5
                )
                if r.status_code != 200:
                    continue

                out = r.json()
                v = float(out.get('linear_x', 0.0))
                w = float(out.get('angular_z', 0.0))

                # ✅ 조향/전진 부호 반전은 여기 한 곳에서만!
                if self.invert_v:
                    v = -v
                if self.invert_w:
                    w = -w

                cmd = Twist()
                cmd.linear.x  = v
                cmd.angular.z = w
                self.pub.publish(cmd)

            except Exception:
                pass

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyHttpClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
