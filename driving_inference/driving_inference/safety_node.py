#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import cv2
import numpy as np
import torch
import json
import os
import threading
import time
from collections import deque
from ultralytics import YOLO
from depth_anything_3.api import DepthAnything3

class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')

        self.cb_group = MutuallyExclusiveCallbackGroup()

        # --- 이미지 리소스 로드 ---
        image_path = 'handlebar.png' 
        if os.path.exists(image_path):
            self.steering_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            scale_percent = 200 / self.steering_img.shape[1]
            w = int(self.steering_img.shape[1] * scale_percent)
            h = int(self.steering_img.shape[0] * scale_percent)
            self.steering_img = cv2.resize(self.steering_img, (w, h))
        else:
            self.steering_img = None

        # --- 설정 변수 ---
        self.SAFE_DIST_DRIVING = 1.5
        self.SAFE_DIST_PARKING = 0.7
        self.SAFE_DIST_REVERSE = 0.5
        self.stop_prob_thresh = 0.2
        
        # [설정] Call 모드 정지 거리 (1.0m)
        self.aruco_stop_dist = 1.3
        
        # --- 상태 변수 ---
        self.current_vel = Twist() 
        self.robot_mode = "call" 
        self.current_stop_dist = self.SAFE_DIST_DRIVING
        
        self.front_queue = deque(maxlen=20)
        self.rear_queue = deque(maxlen=20)
        self.is_front_danger = False
        self.is_rear_danger = False
        self.is_danger = False 
        self.is_arrived = False
        self.is_parking_complete = False

        self.latest_front_msg = None
        self.latest_rear_msg = None
        
        self.front_res_img = np.zeros((240, 320, 3), dtype=np.uint8)
        self.rear_res_img = np.zeros((240, 320, 3), dtype=np.uint8)

        # --- AI 모델 로드 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO('yolo11n.pt')
        self.get_logger().info("YOLO Warming up...")
        self.yolo(np.zeros((240, 320, 3), dtype=np.uint8), verbose=False)
        
        try:
            self.depth = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(self.device)
        except:
            self.get_logger().error("Depth Load Failed")
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.cam_mtx = np.array([[551.95, 0, 309.32], [0, 551.02, 233.54], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.492, 0.222, -0.007, -0.001, 0.0], dtype=np.float32)

        # --- ROS Communication ---
        self.sub_front_img = self.create_subscription(CompressedImage, '/front_cam/image/compressed', self.front_img_callback, 1, callback_group=self.cb_group)
        self.sub_rear_img = self.create_subscription(CompressedImage, '/rear_cam/image/compressed', self.rear_img_callback, 1, callback_group=self.cb_group)
        self.sub_mode = self.create_subscription(String, '/robot_mode', self.mode_callback, 10, callback_group=self.cb_group)
        self.sub_parking_complete = self.create_subscription(Bool, '/parking/complete', self.parking_complete_callback, 10, callback_group=self.cb_group)
        self.sub_cmd_vel = self.create_subscription(Twist, '/controller/cmd_vel', self.vel_callback, 10, callback_group=self.cb_group)

        # [Safety Output] 비상 정지 명령 (Priority 100)
        self.pub_stop_cmd = self.create_publisher(Twist, '/safety/stop_cmd', 10)
        self.pub_res_img = self.create_publisher(CompressedImage, '/safety/result/compressed', 1)
        self.pub_status = self.create_publisher(String, '/robot_status', 10)

        # AI 스레드 시작
        self.ai_thread = threading.Thread(target=self.ai_processing_loop)
        self.ai_thread.daemon = True
        self.ai_thread.start()

    # =========================================================
    # 1. ROS Callbacks
    # =========================================================
    def front_img_callback(self, msg):
        self.latest_front_msg = msg

    def rear_img_callback(self, msg):
        if self.robot_mode == "parking":
            self.latest_rear_msg = msg
        else:
            self.latest_rear_msg = None

    def vel_callback(self, msg): 
        self.current_vel = msg

    def mode_callback(self, msg):
        self.robot_mode = msg.data
        if self.robot_mode != "parking":
            self.is_rear_danger = False
            self.rear_queue.clear()
            self.latest_rear_msg = None
        
        if self.robot_mode == "parking":
            self.current_stop_dist = self.SAFE_DIST_PARKING
            self.is_parking_complete = False
        else:
            self.current_stop_dist = self.SAFE_DIST_DRIVING
    
    def parking_complete_callback(self, msg):
        if self.robot_mode == "parking" and msg.data is True:
            self.is_parking_complete = True

    # =========================================================
    # 2. AI Processing Loop
    # =========================================================
    def ai_processing_loop(self):
        while rclpy.ok():
            try:
                current_front_msg = self.latest_front_msg
                current_rear_msg = self.latest_rear_msg
                
                if current_front_msg is None:
                    time.sleep(0.01)
                    continue

                # --- 전방 처리 ---
                np_arr = np.frombuffer(current_front_msg.data, np.uint8)
                img_front = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img_front is not None:
                    h, w = img_front.shape[:2]
                    infer_img = cv2.resize(img_front, (320, 240))
                    
                    with torch.no_grad():
                        depth_small = self.depth.inference([infer_img]).depth[0]
                    depth_map = cv2.resize(depth_small, (w, h))
                    
                    self.check_danger(img_front, depth_map, is_rear=False)
                    
                    # [UI] Aruco 감지 및 정보 표시
                    self.check_aruco(img_front)
                    
                    vel_text = f"L:{self.current_vel.linear.x:.2f} A:{self.current_vel.angular.z:.2f}"
                    cv2.putText(img_front, vel_text, (10, h-20), 0, 0.5, (255,255,255), 1)
                    img_front = self.overlay_steering(img_front, self.steering_img, self.current_vel.angular.z)
                    cv2.putText(img_front, "[FRONT]", (10, 30), 0, 0.7, (0, 255, 0), 2)
                    self.front_res_img = img_front

                # --- 후방 처리 ---
                if self.robot_mode == "parking" and current_rear_msg is not None:
                    np_arr_r = np.frombuffer(current_rear_msg.data, np.uint8)
                    img_rear = cv2.imdecode(np_arr_r, cv2.IMREAD_COLOR)
                    
                    if img_rear is not None:
                        h_r, w_r = img_rear.shape[:2]
                        infer_img_r = cv2.resize(img_rear, (320, 240))
                        
                        with torch.no_grad():
                            depth_small_r = self.depth.inference([infer_img_r]).depth[0]
                        depth_map_r = cv2.resize(depth_small_r, (w_r, h_r))
                        
                        self.check_danger(img_rear, depth_map_r, is_rear=True)
                        cv2.putText(img_rear, "[REAR]", (10, 30), 0, 0.7, (0, 0, 255), 2)
                        self.rear_res_img = img_rear

                self.update_safety_state()

                # --- [핵심] TwistMux 제어 로직 ---
                should_stop = (
                    self.is_danger or 
                    self.robot_mode == "stop" or 
                    (self.robot_mode == "call" and self.is_arrived) # <-- Call 모드 도착 시 정지
                )

                if should_stop:
                    # TwistMux Priority 100번 점유 -> 강제 정지
                    stop_msg = Twist()
                    self.pub_stop_cmd.publish(stop_msg)
                    
                    # 도착 로그 출력
                    # if self.robot_mode == "call" and self.is_arrived:
                    #     self.get_logger().info("Call Mode: Target Arrived. Stopping.", throttle_duration_sec=2.0)

                self.publish_result_image()

            except Exception as e:
                self.get_logger().error(f"AI Loop Error: {e}")
                time.sleep(0.1)

    # ... (기타 함수들) ...
    def update_safety_state(self):
        if self.is_front_danger or self.is_rear_danger: self.is_danger = True
        else: self.is_danger = False
        
        status_text = "SAFE"
        if self.robot_mode == "stop": status_text = "SYSTEM STOPPED"
        elif self.is_danger: status_text = "DANGER STOP"
        elif self.robot_mode == "call" and self.is_arrived: status_text = "ARRIVED STOP"
        elif self.robot_mode == "parking" and self.is_parking_complete: status_text = "PARKING COMPLETE"
        
        data = {"mode": self.robot_mode, "status": status_text}
        msg = String(); msg.data = json.dumps(data)
        self.pub_status.publish(msg)

    def publish_result_image(self):
        try:
            final_img = None
            if self.robot_mode == "parking":
                h_f, w_f = self.front_res_img.shape[:2]
                h_r, w_r = self.rear_res_img.shape[:2]
                if h_r > 0 and w_r > 0:
                    scale = w_f / w_r
                    new_h = int(h_r * scale)
                    rear_resized = cv2.resize(self.rear_res_img, (w_f, new_h))
                    final_img = np.vstack((self.front_res_img, rear_resized))
                else:
                    final_img = self.front_res_img
            else:
                final_img = self.front_res_img

            if final_img is None: return

            status_color = (0, 0, 255) if self.is_danger else (0, 255, 0)
            status_str = "DANGER" if self.is_danger else f"{self.robot_mode.upper()}"
            
            # [도착 상태 표시]
            if self.robot_mode == "call" and self.is_arrived:
                status_str = "ARRIVED"
                status_color = (0, 255, 255) # 노란색

            text_size = cv2.getTextSize(status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (final_img.shape[1] - text_size[0]) // 2
            
            cv2.putText(final_img, f"{time.time() % 100:.2f}", (final_img.shape[1]-80, 30), 0, 0.6, (255,255,0), 2)
            cv2.putText(final_img, status_str, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            out_msg = CompressedImage()
            out_msg.format = "jpeg"
            out_msg.data = np.array(cv2.imencode('.jpg', final_img)[1]).tobytes()
            self.pub_res_img.publish(out_msg)
        except Exception as e:
            pass

    def check_aruco(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        self.is_arrived = False # 기본값 리셋

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, self.cam_mtx, self.dist_coeffs)
            min_dist = 999.0
            
            for i in range(len(ids)):
                dist = tvecs[i][0][2] # Z축 거리
                if dist < min_dist:
                    min_dist = dist
                cv2.aruco.drawDetectedMarkers(img, corners)
            
            # [화면 표시] 감지된 최단 거리 표시
            dist_text = f"Marker: {min_dist:.2f}m"
            cv2.putText(img, dist_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # [핵심 로직] 1.0m 이내이면 도착 처리
            if min_dist <= self.aruco_stop_dist:
                self.is_arrived = True
                cv2.putText(img, "STOP AREA", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def check_danger(self, img, depth_map, is_rear):
        results = self.yolo(img, verbose=False, conf=0.5)
        danger_count = 0
        h, w = img.shape[:2]
        cx = w // 2
        margin = int(w * 0.25)
        
        if is_rear: check_dist = self.SAFE_DIST_REVERSE
        else: check_dist = self.current_stop_dist

        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = self.yolo.names[cls]
            if name not in ['person', 'car', 'truck', 'motorcycle']: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = depth_map[y1:y2, x1:x2]
            dist = np.median(roi) if roi.size > 0 else 99.9
            box_cx = (x1 + x2) // 2
            in_path = (cx - margin < box_cx < cx + margin)
            color = (0, 255, 0)
            if in_path and dist < check_dist:
                danger_count = 1
                color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {dist:.1f}m", (x1, y1-5), 0, 0.5, color, 2)
            
        if is_rear:
            self.rear_queue.append(danger_count)
            self.is_rear_danger = (sum(self.rear_queue) / len(self.rear_queue) if self.rear_queue else 0.0) >= self.stop_prob_thresh
        else:
            self.front_queue.append(danger_count)
            self.is_front_danger = (sum(self.front_queue) / len(self.front_queue) if self.front_queue else 0.0) >= self.stop_prob_thresh

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    def overlay_steering(self, background, steering_wheel, angle_deg):
        if steering_wheel is None: return background
        rotated = self.rotate_image(steering_wheel, angle_deg * 10)
        bh, bw = background.shape[:2]
        sh, sw = rotated.shape[:2]
        x_offset = (bw - sw) // 2
        y_offset = bh - sh - 30
        roi = background[y_offset:y_offset+sh, x_offset:x_offset+sw]
        if rotated.shape[2] == 4:
            alpha = rotated[:, :, 3] / 255.0
            for c in range(0, 3):
                roi[:, :, c] = (alpha * rotated[:, :, c] + (1 - alpha) * background[y_offset:y_offset+sh, x_offset:x_offset+sw, c])
        else:
            cv2.addWeighted(roi, 0.5, rotated[:,:,:3], 0.5, 0, roi)
        return background

def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()