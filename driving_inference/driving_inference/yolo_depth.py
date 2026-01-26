import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from depth_anything_3.api import DepthAnything3

class YoloDepthQueueNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_queue_node')

        # --- [1. 설정 파라미터] ---
        self.conf_threshold = 0.5       # YOLO 탐지 신뢰도
        self.stop_distance = 1.5        # 정지 거리 (1.5m 이내면 위험)
        self.center_ratio = 0.5         # 화면 중앙 50% 영역 감지
        
        # 큐 설정 (노이즈 필터링용)
        self.queue_size = 20
        self.stop_prob_threshold = 0.6  # 60% 이상 위험 감지 시 정지
        self.stop_queue = deque(maxlen=self.queue_size)

        # 모델 ID
        self.DEPTH_MODEL_ID = "depth-anything/DA3METRIC-LARGE" 

        # --- [2. 모델 로드] ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # YOLO 로드
        self.get_logger().info('Loading YOLO11n model...')
        self.yolo_model = YOLO('yolo11n.pt')

        # Depth Anything 로드
        self.get_logger().info(f'Loading Depth Anything model ({self.device})...')
        try:
            self.depth_model = DepthAnything3.from_pretrained(self.DEPTH_MODEL_ID)
            self.depth_model = self.depth_model.to(device=self.device)
        except Exception as e:
            self.get_logger().error(f"Depth 모델 로드 실패: {e}")
            exit()

        self.bridge = CvBridge()

        # --- [3. 통신 설정] ---
        self.sub = self.create_subscription(
            CompressedImage,
            '/front_cam/image/compressed',
            self.image_callback,
            1
        )
        self.res_pub = self.create_publisher(
            CompressedImage,
            '/yolo_depth/result/compressed',
            1
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            '/controller/cmd_vel',
            10
        )

    def image_callback(self, msg):
        try:
            # 1. 이미지 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            h, w, _ = cv_image.shape

            # 2. Depth Anything 추론 (거리 맵 생성)
            with torch.no_grad():
                prediction = self.depth_model.inference([cv_image])
            depth_map = prediction.depth[0]  # 단위: 미터(m)

            # 3. YOLO 추론 (객체 탐지)
            results = self.yolo_model(cv_image, conf=self.conf_threshold, verbose=False)
            boxes = results[0].boxes

            # --- [4. 영역 계산 및 시각화] ---
            center_x = w // 2
            margin = int(w * self.center_ratio / 2)
            roi_x1 = center_x - margin
            roi_x2 = center_x + margin
            
            # 중앙 영역 가이드라인 (노란색 점선)
            cv2.line(cv_image, (roi_x1, 0), (roi_x1, h), (0, 255, 255), 1)
            cv2.line(cv_image, (roi_x2, 0), (roi_x2, h), (0, 255, 255), 1)

            # 현재 프레임의 위험 여부 판단 (0: 안전, 1: 위험)
            frame_danger_signal = 0 
            min_dist_in_frame = 999.0

            # 5. 객체 분석 루프
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]

                # 화면 밖 좌표 클리핑
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if class_name in ['person', 'motorcycle', 'car', 'truck']:
                    # A. 거리 측정: 박스 영역의 Depth 값 추출
                    person_depth_roi = depth_map[y1:y2, x1:x2]
                    
                    if person_depth_roi.size > 0:
                        # 중앙값(median)을 사용하여 노이즈나 배경 영향을 줄임
                        dist = np.median(person_depth_roi)
                    else:
                        dist = 999.0

                    # B. 위치 판단
                    box_cx = (x1 + x2) // 2
                    is_in_center = (roi_x1 < box_cx < roi_x2)
                    is_too_close = (dist < self.stop_distance)

                    # C. 시각화 및 신호 처리
                    color = (0, 255, 0) # 기본: 초록
                    thickness = 2

                    if is_in_center:
                        if is_too_close:
                            # 위험 상황! (중앙 + 가까움)
                            frame_danger_signal = 1
                            min_dist_in_frame = min(min_dist_in_frame, dist)
                            color = (0, 0, 255) # 빨강
                            thickness = 3
                        else:
                            # 중앙이지만 안전 거리 확보됨
                            color = (0, 255, 255) # 노랑
                    
                    # 박스 및 거리 텍스트 그리기
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness)
                    label = f"{class_name} {dist:.1f}m"
                    cv2.putText(cv_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- [6. 큐(Queue) 업데이트 및 확률 계산] ---
            self.stop_queue.append(frame_danger_signal)
            
            # 현재 큐의 위험 확률 계산 (예: 20개 중 15개가 위험이면 0.75)
            if len(self.stop_queue) > 0:
                current_prob = sum(self.stop_queue) / len(self.stop_queue)
            else:
                current_prob = 0.0

            # --- [7. 로봇 제어 및 UI 표시] ---
            # 확률 바(Bar) 그리기
            bar_w = 200
            bar_h = 20
            cv2.rectangle(cv_image, (10, 10), (10 + bar_w, 10 + bar_h), (50, 50, 50), -1) # 배경
            
            fill_w = int(bar_w * current_prob)
            status_color = (0, 255, 0) # 초록
            
            if current_prob >= self.stop_prob_threshold:
                status_color = (0, 0, 255) # 빨강
                # 정지 명령
                self.stop_robot()
                warning_text = f"STOP! Risk: {current_prob*100:.0f}%"
                cv2.putText(cv_image, warning_text, (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                self.get_logger().warn(f"Stopping... Risk: {current_prob:.2f}, Dist: {min_dist_in_frame:.2f}m", throttle_duration_sec=1)
            else:
                # 안전 상태 텍스트
                cv2.putText(cv_image, f"Risk: {current_prob*100:.1f}%", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 게이지 채우기
            cv2.rectangle(cv_image, (10, 10), (10 + fill_w, 10 + bar_h), status_color, -1)

            # 8. 결과 이미지 발행
            success, encoded_img = cv2.imencode('.jpg', cv_image)
            if success:
                out_msg = CompressedImage()
                out_msg.format = "jpeg"
                out_msg.data = np.array(encoded_img).tobytes()
                self.res_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthQueueNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()