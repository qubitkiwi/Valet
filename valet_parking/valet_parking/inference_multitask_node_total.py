# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from typing import Dict, Tuple, Optional

# ★ 저장해둔 모델 파일명과 Class 이름이 맞는지 꼭 확인하세요!
from .mobilenetv3s_parking_model_pretrained_multi import MultiCamParkingModel



# -------------------------
# 설정 (학습 코드와 동일하게 맞춰야 함)
# -------------------------
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 2  # ★ 학습할 때 설정한 클래스 개수

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
        """
        Args:
            stop_threshold (float): 정지를 트리거할 확률 임계값 (기본 0.9 = 90%) 
        """
        self.stop_threshold = stop_threshold

        self.is_parking_finished = False

    def apply_safety_logic(self, linear_x: float, angular_z: float, prob_complete: float):
        """
        주차 완료 확률(prob_complete)이 임계값을 넘으면 속도를 0으로 만듭니다.
        
        Returns:
            linear_x, angular_z, is_stopped (bool)
        """
        # 1. 상태 업데이트: 아직 안 끝났는데, 이번에 확률이 기준을 넘었다면 '완료' 상태로 변경
        if not self.is_parking_finished:
            if prob_complete >= self.stop_threshold:
                self.is_parking_finished = True

        # 2. 제어 적용: 완료 상태라면 무조건 0 출력
        if self.is_parking_finished:
            return 0.0, 0.0, True  # (속도, 조향, 정지상태여부)
            
        # 완료 상태가 아니라면 원래 값 그대로 반환
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
    """
    학습 데이터셋의 전처리 방식과 동일하게 수행
    1. BGR -> RGB
    2. Crop
    3. Resize
    4. Transpose (C, H, W)
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Crop 적용
    if cam_name in CROP_SETTINGS:
        y1, y2, x1, x2 = CROP_SETTINGS[cam_name]
        h, w, _ = img.shape
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(h, y2); x2 = min(w, x2)
        
        if y2 > y1 and x2 > x1:
            img = img[y1:y2, x1:x2]

    # 3. Resize
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # 4. (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    # Float32 변환 (Normalization은 모델 내부에서 수행됨)
    return img.astype(np.float32)


# -----------------------------------------------------------
# .Main Node Class
# -----------------------------------------------------------
class MultiTaskInferNode(Node):
    def __init__(self):
        super().__init__('multitask_infer_node')

        # 파라미터 선언
        self.declare_parameter('cams', ['front_cam', 'rear_cam', 'left_cam', 'right_cam'])
        self.declare_parameter('ckpt_path', '/home/hyunii/everything/parking/multi/mobilenetv3s_pretrained_up_mid_crop_LR_cls_04_sampler_onecycle_batch256_epoch100_lr0001/best_model.pth') # 경로 수정 필요
        self.declare_parameter('linear_gain', 1.0) # 속도 배율
        self.declare_parameter('angular_gain', 1.0) # 조향 배율
        

        self.cams = list(self.get_parameter('cams').value)
        self.ckpt_path = self.get_parameter('ckpt_path').value
        self.linear_gain = self.get_parameter('linear_gain').value
        self.angular_gain = self.get_parameter('angular_gain').value

        # ★ [수정] 안전 제어 컨트롤러 초기화 ===================================================================================
        self.declare_parameter('stop_prob_threshold', 0.9)
        stop_threshold = self.get_parameter('stop_prob_threshold').value

        self.safety_controller = ParkingSafetyController(stop_threshold=stop_threshold)
        self.get_logger().info(f"🛡️ Safety Controller Active (Stop Threshold: {stop_threshold*100:.1f}%)")
        # ==================================================================================================================

        # 동기화 설정
        self.sync_slop = 0.1
        self.pub_hz = 10.0

        # Device 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"🚀 Using Device: {self.device}")

        # ---------------------------
        # 모델 로드 (Multi-Task)
        # ---------------------------
        self.get_logger().info("⏳ Loading Model...")
        # ★ 학습 시 사용한 num_classes와 동일해야 함
        self.model = MultiCamParkingModel(pretrained=True, num_classes=NUM_CLASSES).to(self.device)
        
        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.get_logger().info(f"✅ Loaded weights from {self.ckpt_path}")
        else:
            self.get_logger().warn("⚠️ No checkpoint path provided!")

        self.model.eval()

        # 이미지 버퍼 초기화
        self.img_buf: Dict[str, Tuple[float, bytes]] = {}

        # Publisher / Subscriber
        self.pub_cmd = self.create_publisher(Twist, '/controller/cmd_vel', 10)

        for cam in self.cams:
            topic = f'/{cam}/image/compressed' # 토픽명 확인 필요 (ex: /front_cam/image/compressed)
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

    def on_img(self, cam: str, msg: CompressedImage):
        if cam not in self.cams: return
        t = stamp_to_sec(msg.header.stamp)
        self.img_buf[cam] = (t, bytes(msg.data))
        
        # 버퍼 관리 (너무 오래된 데이터 삭제)
        if len(self.img_buf) > 16:
            oldest = min(self.img_buf.keys(), key=lambda k: self.img_buf[k][0])
            del self.img_buf[oldest]

    def _pop_synced(self) -> Optional[Dict[str, bytes]]:
        # 4개 카메라 데이터가 모두 있는지 확인
        for cam in self.cams:
            if cam not in self.img_buf: return None
            
        times = [self.img_buf[cam][0] for cam in self.cams]
        if (max(times) - min(times)) > self.sync_slop:
            # 싱크 안 맞으면 가장 오래된 것 버림
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

        # 1. 전처리 (4개 이미지 -> Tensor)
        images_list = []
        for cam in self.cams:
            bgr = decode_jpeg_to_bgr(synced[cam])
            if bgr is None: return
            img_tensor = preprocess_like_train(bgr, cam) # (C,H,W)
            images_list.append(img_tensor)
        
        # Stack -> (1, 4, 3, 224, 224)
        x_np = np.stack(images_list, axis=0)
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self.device)

        # 2. 모델 추론
        outputs = self.model(x_tensor)
        
        # 3. 결과 파싱
        # (1) 제어값 (Regression)
        control_out = outputs['control'] # (B, 2)
        linear_x = control_out[0, 0].item() * self.linear_gain
        angular_z = control_out[0, 1].item() * self.angular_gain
        
        # (2) 분류값 (Classification)
        class_out = outputs['class']     # (B, num_classes)
        probs = F.softmax(class_out, dim=1)
        p0 = probs[0, 0].item()
        p1 = probs[0,1].item()
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item() * 100.0
        
        pred_str = CLASS_NAMES.get(pred_idx, f"Unknown({pred_idx})")

        # -----------------------------------------------------------
        # ★ [수정] 모듈화된 안전 로직 적용
        #   (한 번 멈추면 계속 0 반환)
        # -----------------------------------------------------------
        linear_x, angular_z, is_stopped = self.safety_controller.apply_safety_logic(
            linear_x, angular_z, p1
        )

        if is_stopped:
        #     # 로그도 확실하게 변경
            pred_str = "🛑 PARKING COMPLETE (Holding STOP)"
        # -----------------------------------------------------------


        # 4. 제어 메시지 발행
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub_cmd.publish(msg)

        # 5. 로깅
        self.get_logger().info(
            f"🚗 V: {linear_x:.3f}, W: {angular_z:.3f} | "
            f"🛑 State: {pred_str} ({confidence:.1f}%) | "
            f"p0={p0:.3f}, p1={p1:.3f}"
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