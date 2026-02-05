#!/usr/bin/env python3
# encoding: utf-8
import os
import sys
import struct
import math
import time
import threading
import rclpy
from enum import Enum
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from std_msgs.msg import Int32
from std_msgs.msg import Bool

from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState


# --- Helper Function ---
def val_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# --- Constants ---
AXES_MAP = ['lx', 'ly', 'rx', 'ry', 'r2', 'l2', 'hat_x', 'hat_y']
BUTTON_MAP = ['Y', 'B', 'A', 'X', 'l1', 'r1', 'l2', 'r2', 'select', 'start', 'l3', 'r3','mode']

class ButtonState(Enum):
    Normal = 0
    Pressed = 1
    Holding = 2
    Released = 3

class DirectJoystickController(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # --- ROS 2 파라미터 ---
        self.declare_parameter('max_linear', 0.7)
        self.declare_parameter('max_angular', 3.0)
        self.declare_parameter('disable_servo_control', True)
        self.declare_parameter('joy_topic', '/joystick/cmd_vel')
        self.declare_parameter('intervention_topic', '/human_intervention_state')

        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value
        self.disable_servo_control = self.get_parameter('disable_servo_control').value
        self.machine = os.environ.get('MACHINE_TYPE', 'MentorPi_Acker')
        self.joy_topic = self.get_parameter('joy_topic').get_parameter_value().string_value

        self.get_logger().info(f'Machine: {self.machine}, Max Lin: {self.max_linear}')

        # --- Publishers ---
        self.mecanum_pub = self.create_publisher(Twist, self.joy_topic, 10)
        self.recording_sub = self.create_subscription(Int32, 'record_control', self.record_control_callback, 10)
        self.intervention_topic = self.get_parameter('intervention_topic').get_parameter_value().string_value

        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        self.buzzer_pub = self.create_publisher(BuzzerState, 'ros_robot_controller/set_buzzer', 10)
        self.recording = self.create_publisher(Int32, 'record_control', 2)

        # --- 상태 변수 ---
        self.min_value = 0.1
        self.last_axes = dict(zip(AXES_MAP, [0.0] * len(AXES_MAP)))
        self.last_buttons = dict(zip(BUTTON_MAP, [0.0] * len(BUTTON_MAP)))

        self.raw_axes = [0.0] * 10
        self.raw_buttons = [0] * 20
        self.recording_started = False
        
        # 사람 개입 상태 변수 (False: AI/자동모드 가능, True: 사람 개입 중 - 로직 정지)
        self.human_pub = self.create_publisher(Bool, self.intervention_topic, 10)
        self.human_intervention = True

        # --- 쓰레드 시작 ---
        self.running = True
        self.read_thread = threading.Thread(target=self.read_joystick_loop)
        self.read_thread.daemon = True
        self.read_thread.start()

        self.get_logger().info('Direct Joystick Controller Started (Raw Mode)')

    def record_control_callback(self, msg):
        mode = msg.data
        if mode != 0:
            self.recording_started = True
        else:
            self.recording_started = False

    def read_joystick_loop(self):
        device_path = "/dev/input/js0"
        if not os.path.exists(device_path):
            self.get_logger().error(f"Device {device_path} not found.")
            return

        try:
            with open(device_path, "rb") as js_file:
                while self.running and rclpy.ok():
                    event_data = js_file.read(8)
                    if not event_data: break
                    time_ms, value, type_, number = struct.unpack("IhBB", event_data)
                    real_type = type_ & ~0x80
                    updated = False

                    if real_type == 1: # Button
                        if number < len(self.raw_buttons):
                            self.raw_buttons[number] = 1 if value else 0
                            updated = True
                    elif real_type == 2: # Axis
                        if number < len(self.raw_axes):
                            norm_val = value / 32767.0
                            if abs(norm_val) < 0.05: norm_val = 0.0
                            self.raw_axes[number] = norm_val
                            updated = True

                    if updated:
                        self.process_control_logic()
        except Exception as e:
            self.get_logger().error(f"Joystick read error: {e}")

    def process_control_logic(self):
        current_axes_dict = dict(zip(AXES_MAP, self.raw_axes[:len(AXES_MAP)]))
        hat_x, hat_y = current_axes_dict.get('hat_x', 0), current_axes_dict.get('hat_y', 0)
        hat_xl, hat_xr = (1 if hat_x > 0.5 else 0), (1 if hat_x < -0.5 else 0)
        hat_yu, hat_yd = (1 if hat_y > 0.5 else 0), (1 if hat_y < -0.5 else 0)
        
        current_buttons_list = self.raw_buttons[:12]
        current_buttons_list.extend([hat_xl, hat_xr, hat_yu, hat_yd, 0])
        current_buttons_dict = dict(zip(BUTTON_MAP, current_buttons_list))

        # Axis 처리
        axes_changed = any(self.last_axes[key] != current_axes_dict.get(key, 0) for key in AXES_MAP)
        if axes_changed:
            self.axes_callback(current_axes_dict)

        # Button 처리
        for key, val in current_buttons_dict.items():
            if not key: continue
            last_val = self.last_buttons.get(key, 0)
            if val != last_val:
                new_state = ButtonState.Pressed if val > 0 else ButtonState.Released
                callback_name = f"{key}_callback"
                if hasattr(self, callback_name):
                    getattr(self, callback_name)(new_state)

        self.last_axes = current_axes_dict
        self.last_buttons = current_buttons_dict

    def axes_callback(self, axes):

        twist = Twist()
        ly = axes['ly'] if abs(axes['ly']) > self.min_value else 0.0
        rx = axes['rx'] if abs(axes['rx']) > self.min_value else 0.0

        twist.linear.x = -val_map(ly, -1, 1, -self.max_linear, self.max_linear)
        steering_angle = -val_map(rx, -1, 1, -math.radians(45), math.radians(45))        
        
        if self.disable_servo_control and self.human_intervention == True:
            servo_state = PWMServoState()
            servo_state.id = [3]
            
            if steering_angle == 0:
                twist.angular.z = 0.0
                servo_state.position = [1500]
            else:
                try:
                    R = 0.145 / math.tan(steering_angle)
                    twist.angular.z = float(twist.linear.x / R)
                except ZeroDivisionError:
                    twist.angular.z = 0.0
                servo_state.position = [1500 + int(math.degrees(steering_angle) / 180 * 2000)]
            
            data = SetPWMServoState()
            data.state = [servo_state]
            data.duration = 0.02
            self.servo_state_pub.publish(data)

        self.mecanum_pub.publish(twist)

    # --- Button Callbacks ---
    def select_callback(self, new_state): 
        if new_state == ButtonState.Pressed:
            # 상태 토글
            self.human_intervention = not self.human_intervention
            state_str = "ENABLED (Logic Stopped)" if self.human_intervention else "DISABLED (Logic Resumed)"
            self.get_logger().info(f'Human Intervention {state_str}')
        

            # 기존 레코딩 관련 토픽 유지
            msg = Bool()
            msg.data = self.human_intervention
            self.human_pub.publish(msg)

    def start_callback(self, new_state):
        if new_state == ButtonState.Pressed:
            msg = BuzzerState()
            msg.freq = 2500
            msg.on_time = 0.05
            msg.off_time = 0.01
            msg.repeat = 1
            self.buzzer_pub.publish(msg)
            self.get_logger().info('Buzzer Beep!')

    def Y_callback(self, new_state): 
        if new_state == ButtonState.Pressed:
            msg = Int32()
            msg.data = 2
            self.recording.publish(msg)

    def B_callback(self, new_state): 
        if new_state == ButtonState.Pressed:
            msg = Int32()
            msg.data = 3
            self.recording.publish(msg)

    def A_callback(self, new_state): 
        if new_state == ButtonState.Pressed:
            msg = Int32()
            msg.data = 0
            self.recording.publish(msg)

    def X_callback(self, new_state): 
        if new_state == ButtonState.Pressed:
            msg = Int32()
            msg.data = 1
            self.recording.publish(msg)

    def l2_callback(self, new_state):
        if new_state == ButtonState.Pressed:
            if self.max_linear - 0.05 > 0: self.max_linear -= 0.05
            self.get_logger().info(f'Max Speed: {self.max_linear:.2f}')

    def r2_callback(self, new_state):
        if new_state == ButtonState.Pressed:
            if self.max_linear + 0.05 <= 1.5: self.max_linear += 0.05
            self.get_logger().info(f'Max Speed: {self.max_linear:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = DirectJoystickController('direct_joystick_ctrl')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()