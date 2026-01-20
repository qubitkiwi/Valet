#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String, Float32, Int32

from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState


class CmdMuxNode(Node):
    """
    DirectJoystickController 방식으로 동작하는 cmd_mux:
    - /mux/select: 'joy' | 'policy' | 'stop'
    - /cmd_vel_joy, /cmd_vel_policy (Twist) 입력
      * linear.x  : 목표 속도
      * angular.z : "조향 입력(스틱 rx)"으로 해석 (요레이트가 아님)

    - 내부에서:
      steer_input -> 조향각 delta(rad) 직접 생성
      w = v * tan(delta) / wheelbase 로 angular.z 계산
      servo PWM도 delta로 직접 생성/발행
    """

    def __init__(self):
        super().__init__('cmd_mux')

        # ====== parameters ======
        self.declare_parameter('default', 'joy')
        self.declare_parameter('publish_hz', 20.0)

        # --- JOY 모드 전용 (개입 즉시성) ---
        self.declare_parameter('joy_out_max_linear', 0.25)

        # 입력 제한(안전 상한)
        self.declare_parameter('max_linear', 0.25)

        # 최종 출력 제한(Policy 쪽)
        self.declare_parameter('out_max_linear', 0.15)

        # 변화율 제한(slew-rate) - v만 유지(원하면 0으로 끄면 됨)
        self.declare_parameter('slew_v', 0.15)
        self.declare_parameter('joy_slew_v', 20.0)

        # 차량/서보 파라미터
        self.declare_parameter('wheelbase', 0.145)
        self.declare_parameter('servo_id', 3)
        self.declare_parameter('servo_center', 1500)
        self.declare_parameter('servo_scale', 2000.0)   # 180deg 기준 스케일
        self.declare_parameter('max_steer_deg', 45.0)
        self.declare_parameter('enable_servo', True)
        self.declare_parameter('servo_duration', 0.05)

        # 디버그/옵션
        self.declare_parameter('debug_log', True)
        self.declare_parameter('invert_steer', True)

        # 핵심: 입력 angular.z를 어떻게 해석할지
        # True: angular.z가 [-1,1] 범위의 "스틱 입력"이라고 가정
        # False: angular.z가 [-max_angular, +max_angular] 범위라고 가정하고 정규화해서 사용
        self.declare_parameter('steer_input_normalized', False)

        # steer_input_normalized=False일 때 사용할 입력 스케일(기존 max_angular 역할)
        self.declare_parameter('steer_input_scale', 3.0)

        # 작은 입력 무시 데드존 (DirectJoystick의 min_value 느낌)
        self.declare_parameter('steer_deadzone', 0.05)

        # ====== read params ======
        self.mode = str(self.get_parameter('default').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)

        self.max_linear = float(self.get_parameter('max_linear').value)
        self.joy_out_max_linear = float(self.get_parameter('joy_out_max_linear').value)
        self.out_max_linear = float(self.get_parameter('out_max_linear').value)

        self.slew_v = float(self.get_parameter('slew_v').value)
        self.joy_slew_v = float(self.get_parameter('joy_slew_v').value)

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.servo_id = int(self.get_parameter('servo_id').value)
        self.servo_center = int(self.get_parameter('servo_center').value)
        self.servo_scale = float(self.get_parameter('servo_scale').value)
        self.max_steer_deg = float(self.get_parameter('max_steer_deg').value)
        self.enable_servo = bool(self.get_parameter('enable_servo').value)
        self.servo_duration = float(self.get_parameter('servo_duration').value)

        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.invert_steer = bool(self.get_parameter('invert_steer').value)

        self.steer_input_normalized = bool(self.get_parameter('steer_input_normalized').value)
        self.steer_input_scale = float(self.get_parameter('steer_input_scale').value)
        self.steer_deadzone = float(self.get_parameter('steer_deadzone').value)

        self.max_steer_rad = math.radians(self.max_steer_deg)

        # ====== filtered output state (v만 유지) ======
        self.f_v = 0.0

        # ====== pubs ======
        self.pub_cmd = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.pub_servo = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.pub_steer_deg = self.create_publisher(Float32, '/mux/debug/steer_deg', 10)
        self.pub_servo_pos = self.create_publisher(Int32,   '/mux/debug/servo_pos', 10)

        # 학습/로깅용 라벨 토픽: v는 slew 적용된 f_v, angular.z는 mux 입력(= steer input) 원본
        # - linear.x  : 실제 적용되는 속도(f_v)
        # - angular.z : 선택된 입력(cmd_in.angular.z) 그대로 (요레이트가 아님!)
        self.pub_label_cmd = self.create_publisher(TwistStamped, '/mux/label_cmd', 10)

        # ====== subs ======
        self.sub_sel = self.create_subscription(String, '/mux/select', self._on_select, 10)
        self.sub_joy = self.create_subscription(Twist, '/cmd_vel_joy', self._on_joy, 10)
        self.sub_pol = self.create_subscription(Twist, '/cmd_vel_policy', self._on_policy, 10)

        # ====== state ======
        self.last_joy = Twist()
        self.last_pol = Twist()
        self.timer = self.create_timer(1.0 / self.publish_hz, self._tick)

        if self.debug_log:
            self.get_logger().info(
                f'[MUX] Direct-steer mode ready | steer_input_normalized={self.steer_input_normalized} '
                f'| enable_servo={self.enable_servo}'
            )

    # ---------- callbacks ----------
    def _on_select(self, msg: String):
        m = (msg.data or '').strip()
        if m in ('joy', 'policy', 'stop') and m != self.mode:
            self.mode = m
            self.f_v = 0.0
            if self.debug_log:
                self.get_logger().info(f'[MUX] select="{self.mode}"')

    def _on_joy(self, msg: Twist):
        self.last_joy = msg

    def _on_policy(self, msg: Twist):
        self.last_pol = msg

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ---------- direct-style steering ----------
    def _steer_input_to_delta(self, steer_in: float) -> float:
        """
        DirectJoystickController의 rx -> steering_angle(±45deg) 역할.
        - 입력 steer_in (Twist.angular.z)을 "스틱 입력"으로 보고 [-1,1]로 정규화
        - deadzone 적용
        - δ = steer_norm * max_steer_rad
        """
        # 1) normalize to [-1,1]
        if self.steer_input_normalized:
            steer_norm = float(steer_in)
        else:
            scale = self.steer_input_scale if abs(self.steer_input_scale) > 1e-6 else 1.0
            steer_norm = float(steer_in) / scale

        steer_norm = self._clamp(steer_norm, -1.0, 1.0)

        # 2) deadzone (작은 노이즈 제거)
        if abs(steer_norm) < self.steer_deadzone:
            steer_norm = 0.0

        # 3) invert
        if self.invert_steer:
            steer_norm *= -1.0

        # 4) map to delta(rad)
        delta = steer_norm * self.max_steer_rad
        delta = self._clamp(delta, -self.max_steer_rad, self.max_steer_rad)
        return delta

    def _delta_to_servo_pos(self, delta_rad: float) -> int:
        """
        DirectJoystickController의:
          pos = center + int(deg(delta)/180 * servo_scale)
        """
        return int(self.servo_center + (math.degrees(delta_rad) / 180.0) * self.servo_scale)

    def _publish_servo(self, delta_rad: float):
        pos = self._delta_to_servo_pos(delta_rad)

        servo = PWMServoState()
        servo.id, servo.position = [self.servo_id], [pos]

        msg = SetPWMServoState()
        msg.state, msg.duration = [servo], float(self.servo_duration)
        self.pub_servo.publish(msg)

        self.pub_steer_deg.publish(Float32(data=math.degrees(delta_rad)))
        self.pub_servo_pos.publish(Int32(data=pos))

    # ---------- main loop ----------
    def _tick(self):
        if self.mode == 'stop':
            self._stop_robot()
            return

        # 1) 입력 소스 선택
        cmd_in = self.last_joy if self.mode == 'joy' else self.last_pol

        # 2) 속도 제한 및(선택) slew
        dt = 1.0 / self.publish_hz

        if self.mode == 'joy':
            out_max_v = self.joy_out_max_linear
            s_v = self.joy_slew_v
        else:
            out_max_v = self.out_max_linear
            s_v = self.slew_v

        v_t = self._clamp(float(cmd_in.linear.x), -out_max_v, out_max_v)

        # v slew-rate (원하면 s_v=0으로 끄면 사실상 direct)
        if s_v <= 0.0:
            self.f_v = v_t
        else:
            self.f_v += self._clamp(v_t - self.f_v, -s_v * dt, s_v * dt)

        # 3) Direct 방식 조향: 입력 angular.z를 "steer input"으로 보고 δ로 변환
        delta = self._steer_input_to_delta(float(cmd_in.angular.z))

        # 4) 서보는 δ로 직접 구동
        if self.enable_servo:
            self._publish_servo(delta)

        # 5) 최종 Twist는 아커만 물리로 w 계산 (DirectJoystickController 방식)
        out = Twist()
        out.linear.x = float(self.f_v)

        if abs(delta) < 1e-5 or abs(out.linear.x) < 1e-5:
            out.angular.z = 0.0
        else:
            out.angular.z = float(out.linear.x * math.tan(delta) / self.wheelbase)

        self.pub_cmd.publish(out)

        # =========================
        # ✅ 학습용 라벨 publish 추가
        # - linear.x  : 실제 적용되는 속도(슬루 적용된 f_v)
        # - angular.z : steer input 원본(cmd_in.angular.z)
        # =========================
        lab = TwistStamped()
        lab.header.stamp = self.get_clock().now().to_msg()
        lab.twist.linear.x = float(self.f_v)
        lab.twist.angular.z = float(cmd_in.angular.z)
        self.pub_label_cmd.publish(lab)


    def _stop_robot(self):
        self.pub_cmd.publish(Twist())
        self.f_v = 0.0
        if self.enable_servo:
            # Center 정렬
            self._publish_servo(0.0)


def main(args=None):
    rclpy.init(args=args)
    node = CmdMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
