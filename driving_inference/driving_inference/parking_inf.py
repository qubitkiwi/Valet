# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String 
from typing import Dict, Tuple, Optional, List

from mobilenetv3s_parking_model_pretrained_multi import MultiCamParkingModel

# -------------------------
# 설정 (학습 코드와 동일하게 맞춰야 함)
# -------------------------
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 2

# ★ 학습 코드의 CROP_SETTINGS와 100% 동일해야 함
CROP_SETTINGS = {
    'front_cam': (0, 480, 0, 640),
    'rear_cam':  (0, 480, 0, 640),
    'left_cam':  (0, 300, 100, 640),
    'right_cam': (0, 300, 0, 540)
}

# 표지판/상태 이름 (로그 출력용)
CLASS_NAMES = {
    0: "주차 중",
    1: "주차 완료"
}

# -----------------------------------------------------------
# ParkingSafetyController 클래스 정의
# -----------------------------------------------------------
class ParkingSafetyController:
    def __init__(self, stop_threshold: float = 0.9):
        self.stop_threshold = stop_threshold
        self.is_parking_finished = False

    def apply_safety_logic(self, linear_x: float, angular_z: float, prob_complete: float):
        if prob_complete >= self.stop_threshold:
            return 0.0, 0.0, True  # (속도, 조향, 정지상태여부)

        return linear_x, angular_z, False

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------
def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def decode_jpeg_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def preprocess_like_train(img_bgr: np.ndarray, cam_name: str) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if cam_name in CROP_SETTINGS:
        y1, y2, x1, x2 = CROP_SETTINGS[cam_name]
        h, w, _ = img.shape
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(h, y2); x2 = min(w, x2)
        
        if y2 > y1 and x2 > x1:
            img = img[y1:y2, x1:x2]

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)


# -----------------------------------------------------------
# .Main Node Class
# -----------------------------------------------------------
class MultiTaskInferNode(Node):
    def __init__(self):
        super().__init__('multitask_infer_node')

        # 파라미터 선언
        self.declare_parameter('cams', ['front_cam', 'rear_cam', 'left_cam', 'right_cam'])
        # self.declare_parameter('ckpt_path', 'parking_mobilenetv3s_pretrained_up_mid_crop_LR_cls_05_sampler_final_onecycle_batch256_epoch100_lr0001_model.pth')
        self.declare_parameter('ckpt_path', 'parking_best.pth')
        

        self.declare_parameter('linear_gain', 1.0)
        self.declare_parameter('angular_gain', 1.0)
        
        self.cams = list(self.get_parameter('cams').value)
        self.ckpt_path = self.get_parameter('ckpt_path').value
        self.linear_gain = self.get_parameter('linear_gain').value
        self.angular_gain = self.get_parameter('angular_gain').value

        # 안전 제어 컨트롤러 초기화
        self.declare_parameter('stop_prob_threshold', 0.9)
        stop_threshold = self.get_parameter('stop_prob_threshold').value

        self.safety_controller = ParkingSafetyController(stop_threshold=stop_threshold)
        self.get_logger().info(f"🛡️ Safety Controller Active (Stop Threshold: {stop_threshold*100:.1f}%)")

        # 동기화 설정
        self.sync_slop = 0.1
        self.pub_hz = 10.0

        # Device 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"🚀 Using Device: {self.device}")

        # 모델 로드
        self.get_logger().info("⏳ Loading Model...")
        self.model = MultiCamParkingModel(pretrained=True, num_classes=NUM_CLASSES).to(self.device)
        
        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.get_logger().info(f"✅ Loaded weights from {self.ckpt_path}")
        else:
            self.get_logger().warn("⚠️ No checkpoint path provided!")

        self.model.eval()

        self.img_buf: Dict[str, Tuple[float, bytes]] = {}

        # Publisher
        self.pub_cmd = self.create_publisher(Twist, '/parking/raw_cmd', 10)
        self.pub_complete = self.create_publisher(Bool, '/parking/complete', 10)

        # ★ [추가] 로봇 상태 구독 및 변수 초기화
        self.robot_mode = "unknown"
        self.create_subscription(String, '/robot_status', self.robot_status_callback, 10)

        # Image Subscribers
        for cam in self.cams:
            topic = f'/{cam}/image/compressed'
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, c=cam: self.on_img(c, msg),
                1
            )

        # 타이머 실행
        period = 1.0 / self.pub_hz
        self.create_timer(period, self.tick)
        self.get_logger().info("✨ Node Ready.")

    # ★ [추가] 로봇 모드 콜백 함수
    def robot_status_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict) and "mode" in data:
                self.robot_mode = data["mode"]
            else:
                self.robot_mode = str(msg.data)
        except json.JSONDecodeError:
            self.robot_mode = str(msg.data)

    def on_img(self, cam: str, msg: CompressedImage):
        if cam not in self.cams: return
        t = stamp_to_sec(msg.header.stamp)
        self.img_buf[cam] = (t, bytes(msg.data))
        
        if len(self.img_buf) > 16:
            oldest = min(self.img_buf.keys(), key=lambda k: self.img_buf[k][0])
            del self.img_buf[oldest]

    def _pop_synced(self) -> Optional[Dict[str, bytes]]:
        for cam in self.cams:
            if cam not in self.img_buf: return None
            
        times = [self.img_buf[cam][0] for cam in self.cams]
        if (max(times) - min(times)) > self.sync_slop:
            oldest = min(self.cams, key=lambda c: self.img_buf[c][0])
            del self.img_buf[oldest]
            return None
            
        out = {cam: self.img_buf[cam][1] for cam in self.cams}
        self.img_buf.clear()
        return out

    @torch.no_grad()
    def tick(self):
        synced = self._pop_synced()
        if synced is None: return

        # 1. 전처리
        images_list = []
        for cam in self.cams:
            bgr = decode_jpeg_to_bgr(synced[cam])
            if bgr is None: return
            img_tensor = preprocess_like_train(bgr, cam)
            images_list.append(img_tensor)
        
        x_np = np.stack(images_list, axis=0)
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self.device)

        # 2. 모델 추론
        outputs = self.model(x_tensor)
        
        # 3. 결과 파싱
        control_out = outputs['control']
        linear_x = control_out[0, 0].item() * self.linear_gain
        angular_z = control_out[0, 1].item() * self.angular_gain
        
        class_out = outputs['class']
        probs = F.softmax(class_out, dim=1)
        p0 = probs[0, 0].item()
        p1 = probs[0, 1].item()
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item() * 100.0
        
        pred_str = CLASS_NAMES.get(pred_idx, f"Unknown({pred_idx})")

        # -----------------------------------------------------------
        # 안전 로직 적용 (is_stopped 획득)
        # -----------------------------------------------------------
        linear_x, angular_z, is_stopped = self.safety_controller.apply_safety_logic(
            linear_x, angular_z, p1
        )

        if is_stopped:
            pred_str = "🛑 PARKING COMPLETE (Holding STOP)"

        # 4. 제어 메시지 발행
        # ★ [수정] 모드가 'parking' 일 때만 Twist 및 Complete 신호 발행
        if self.robot_mode == "parking":
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
            self.pub_cmd.publish(msg)

            # 주차 완료 Bool 메시지 발행
            complete_msg = Bool()
            complete_msg.data = is_stopped
            self.pub_complete.publish(complete_msg)

        # 5. 로깅 (모드 상관없이 디버깅용으로 계속 출력)
        self.get_logger().info(
            f"Mode: {self.robot_mode} | "
            f"🚗 V: {linear_x:.3f}, W: {angular_z:.3f} | "
            f"🛑 State: {pred_str} ({confidence:.1f}%) | "
            f"Done: {is_stopped}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MultiTaskInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()