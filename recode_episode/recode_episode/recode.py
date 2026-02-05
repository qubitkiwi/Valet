import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import BuzzerState
from cv_bridge import CvBridge
import cv2
import os
import csv
import numpy as np
from datetime import datetime
from std_msgs.msg import Int32

class DataCollectorService(Node):
    def __init__(self):
        super().__init__('data_collect_service')

        # ===== 설정 (데이터 저장 경로) =====
        self.base_data_dir = os.path.join(os.getcwd(), "collected_data")
        self.base_dir = None
        self.img_dir = None
        self.csv_file = None
        self.csv_writer = None

        self.save_hz = 10.0
        # ===================================

        self.bridge = CvBridge()
        
        # [수정] 4개의 카메라 토픽 설정
        self.camera_names = ['front', 'rear', 'left', 'right']
        self.camera_topics = [
            '/front_cam/image/compressed',
            '/rear_cam/image/compressed',
            '/left_cam/image/compressed',
            '/right_cam/image/compressed'
        ]
        
        # 최신 이미지를 저장할 딕셔너리 (인덱스: 이미지)
        self.latest_images = {i: None for i in range(len(self.camera_topics))}

        self.current_v = 0.0
        self.current_w = 0.0
        self.recording_started = False
        self.parking_mode = 0 

        # 1. 제어 토픽 구독
        self.recording_sub = self.create_subscription(Int32, 'record_control', self.record_control_callback, 10)

        # 2. 카메라 토픽 구독 (루프를 통해 4개 생성)
        for idx, topic in enumerate(self.camera_topics):
            self.create_subscription(
                CompressedImage, 
                topic, 
                lambda msg, i=idx: self.img_callback(msg, i), 
                1
            )
            
        self.cmd_sub = self.create_subscription(Twist, '/controller/cmd_vel',  self.cmd_callback, 10)
        self.buzzer_pub = self.create_publisher(BuzzerState, 'ros_robot_controller/set_buzzer', 1)

        # 3. 타이머
        self.timer = self.create_timer(1.0 / self.save_hz, self.timer_callback)
        
        self.get_logger().info(f"🚀 데이터 수집 노드 준비 완료 (4 Cams). 경로: {self.base_data_dir}")

    def record_control_callback(self, msg):
        mode = msg.data
        if mode == 0:
            if self.recording_started:
                self.recording_started = False
                self.get_logger().info(">>> [명령 수신] 녹화 중지")
                self.play_buzzer(2000)
        else:
            if not self.recording_started:
                self.recording_started = True
                self.parking_mode = mode
                
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_dir = os.path.join(self.base_data_dir, current_time)
                self.img_dir = os.path.join(self.base_dir, "images")
                os.makedirs(self.img_dir, exist_ok=True)
                
                self.csv_path = os.path.join(self.base_dir, "data.csv")
                self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                
                # [수정] CSV 헤더: 4개의 카메라 이미지 경로 포함
                header = ['timestamp'] + [f'{name}_img' for name in self.camera_names] + ['linear_x', 'angular_z']
                self.csv_writer.writerow(header)
                self.csv_file.flush()
                
                self.get_logger().info(f">>> [명령 수신] 녹화 시작 (Mode: {mode})")
                self.play_buzzer(3000)

    def play_buzzer(self, freq):
        buzzer_msg = BuzzerState()
        buzzer_msg.freq = freq
        buzzer_msg.on_time = 0.1
        buzzer_msg.off_time = 0.01
        buzzer_msg.repeat = 1
        self.buzzer_pub.publish(buzzer_msg)

    def img_callback(self, msg, cam_idx):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_images[cam_idx] = img
        except Exception as e:
            self.get_logger().error(f"이미지 {cam_idx} 변환 실패: {e}")

    def cmd_callback(self, msg):
        self.current_v = msg.linear.x
        self.current_w = msg.angular.z
    
    def timer_callback(self):
        if not self.recording_started:
            return

        # 모든 카메라의 이미지가 들어왔는지 확인 (동기화 보장 시도)
        if any(self.latest_images[i] is None for i in range(len(self.camera_topics))):
            return

        try:
            timestamp_str = datetime.now().strftime("%H%M%S_%f")
            saved_filenames = []

            # [수정] 4개 이미지 각각 저장
            for i, name in enumerate(self.camera_names):
                filename = f"images/{name}_{timestamp_str}.jpg"
                save_path = os.path.join(self.base_dir, filename)
                cv2.imwrite(save_path, self.latest_images[i])
                saved_filenames.append(filename)
                
                # 다음 프레임을 위해 초기화 (선택 사항: 동기화 엄격도를 높이려면 필요)
                self.latest_images[i] = None

            # [수정] CSV 행 작성: timestamp, front, back, left, right, v, w
            csv_row = [timestamp_str] + saved_filenames + [self.current_v, self.current_w]
            
            self.csv_writer.writerow(csv_row)
            self.csv_file.flush()

            self.get_logger().info(f"[저장] {timestamp_str} (4 Cams), v={self.current_v}")

        except Exception as e:
            self.get_logger().error(f"저장 중 에러: {e}")

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = DataCollectorService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()