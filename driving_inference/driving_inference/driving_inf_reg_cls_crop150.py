#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import torch
import numpy as np
import cv2
import json

from mobilenet3_multi import MultiTaskDrivingModel


class MobileNetInferenceNode(Node):
    def __init__(self):
        super().__init__('mobilenet_inference_node')

        # --------------------
        # Params
        # --------------------
        self.declare_parameter('MODEL_PATH', './total_data_final_v2_crop150_reg_cls_best1_model.pth')
        self.declare_parameter('input_video', '/front_cam/image/compressed')
        # self.declare_parameter('output_cmd', '/controller/cmd_vel')
        self.declare_parameter('output_cmd', '/driving/raw_cmd')
        
        # ✅ Robot Status Topic 파라미터 추가
        self.declare_parameter('robot_status_topic', '/robot_status')

        self.declare_parameter('LINEAR_GAIN', 1.0)
        self.declare_parameter('STEERING_GAIN', 1.0)

        # ✅ 전처리 옵션
        self.declare_parameter('USE_IMAGENET_NORM', True)

        # (선택) 디버그 오버레이 창 띄우기
        self.declare_parameter('SHOW_DEBUG', False)

        self.MODEL_PATH = self.get_parameter('MODEL_PATH').get_parameter_value().string_value
        self.input_video = self.get_parameter('input_video').get_parameter_value().string_value
        self.output_cmd = self.get_parameter('output_cmd').get_parameter_value().string_value
        self.robot_status_topic = self.get_parameter('robot_status_topic').get_parameter_value().string_value

        self.LINEAR_GAIN = self.get_parameter('LINEAR_GAIN').get_parameter_value().double_value
        self.STEERING_GAIN = self.get_parameter('STEERING_GAIN').get_parameter_value().double_value

        self.USE_IMAGENET_NORM = self.get_parameter('USE_IMAGENET_NORM').get_parameter_value().bool_value
        self.SHOW_DEBUG = self.get_parameter('SHOW_DEBUG').get_parameter_value().bool_value

        # ✅ 현재 로봇 상태 변수 초기화
        self.current_robot_status = "unknown"

        # --------------------
        # ROS I/O
        # --------------------
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.input_video,
            self.image_callback,
            1
        )

        # ✅ Robot Status Subscriber 추가
        self.status_sub = self.create_subscription(
            String,
            self.robot_status_topic,
            self.status_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.output_cmd,
            10
        )

        # --------------------
        # Model
        # --------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.num_signs = self.load_model_from_ckpt(self.MODEL_PATH)

        self.get_logger().info(
            f"✅ Inference node ready | device={self.device} | sub={self.input_video} | pub={self.output_cmd} | num_signs={self.num_signs}"
        )

        # 클래스 이름 매핑
        self.sign_names = {
            0: "일반 주행",
            1: "직진",
            2: "우회전",
            3: "주차",
        }

        # ImageNet normalization stats (torchvision default)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_model_from_ckpt(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            self.get_logger().warn(f"⚠️ MODEL_PATH not found: {ckpt_path}")
            raise FileNotFoundError(f"MODEL_PATH not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # ✅ 학습 코드 저장 포맷: {"model": state_dict, "num_signs": ...}
        if isinstance(ckpt, dict) and ("model" in ckpt):
            state_dict = ckpt["model"]
            num_signs = int(ckpt.get("num_signs", 3))
        else:
            state_dict = ckpt
            num_signs = 3

        model = MultiTaskDrivingModel(num_signs=num_signs)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self.get_logger().info(f"✅ Model loaded: {ckpt_path}")
        return model, num_signs

    # ✅ Robot Status 콜백 함수 추가
    def status_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.current_robot_status = data.get("mode", "unknown")
        except Exception as e:
            self.get_logger().warn(f"Failed to parse robot status: {e}")

    def image_callback(self, msg: CompressedImage):
        # ✅ 조건 체크: driving 또는 call 상태가 아니면 리턴 (Twist 발행 안 함)
        if self.current_robot_status not in ["driving", "call"]:
            return

        bgr = self.decode_compressed(msg)
        if bgr is None:
            self.get_logger().warn("cv2.imdecode failed (image is None)")
            return
        
        # 추론 수행
        v, w, sign_id, sign_prob = self.run_inference(bgr)

        # publish cmd_vel
        twist = Twist()
        twist.linear.x = float(v) * float(self.LINEAR_GAIN)
        twist.angular.z = float(w) * float(self.STEERING_GAIN)
        self.cmd_pub.publish(twist)

        sign_msg = self.sign_names.get(sign_id, f"Class {sign_id}")
        self.get_logger().info(
            f"[{self.current_robot_status}] v={v:.4f}, w={w:.4f} | sign={sign_msg}({sign_id}) p={sign_prob:.2f}"
        )

        if self.SHOW_DEBUG:
            dbg = bgr.copy()
            # 원본 이미지에 텍스트 표시
            cv2.putText(dbg, f"Status: {self.current_robot_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(dbg, f"v={v:.3f} w={w:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(dbg, f"sign={sign_msg}({sign_id}) p={sign_prob:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Crop 영역 표시 (빨간 줄)
            cv2.line(dbg, (0, 150), (dbg.shape[1], 150), (0, 0, 255), 2)
            
            cv2.imshow("mobilenet_infer_crop", dbg)
            cv2.waitKey(1)

    @staticmethod
    def decode_compressed(msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def run_inference(self, bgr_img: np.ndarray):
        # 전처리 (Crop -> Resize -> Normalize)
        x = self.preprocess(bgr_img)

        with torch.no_grad():
            out = self.model(x)

        # out["control"]: [1,2] -> [linear_x, angular_z]
        ctrl = out["control"][0]
        v = float(ctrl[0].item())
        w = float(ctrl[1].item())

        # out["signs"]: logits [1,C]
        logits = out["signs"]
        probs = torch.softmax(logits, dim=1)
        sign_id = int(torch.argmax(probs, dim=1).item())
        sign_prob = float(probs[0, sign_id].item())

        return v, w, sign_id, sign_prob

    def preprocess(self, bgr_img: np.ndarray):
        """
        ✅ 학습 코드(CroppedMultiTaskDataset)와 전처리 과정 통일
        1. BGR -> RGB
        2. Crop Top 150 (img[150:, :])
        3. Resize (224, 224)
        4. Normalize
        """
        # 1. BGR -> RGB
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # [핵심 수정] 상단 150 픽셀 자르기 (Crop Top 150)
        # ---------------------------------------------------------
        # 이미지가 150px보다 큰지 확인
        if rgb.shape[0] > 150:
            rgb = rgb[150:, :]
        else:
            # 만약 이미지가 너무 작으면 예외처리 혹은 그대로 둠
            pass
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # [핵심 수정] Resize (학습 시 224x224로 리사이즈 했으므로 필수)
        # ---------------------------------------------------------
        rgb = cv2.resize(rgb, (224, 224))
        # ---------------------------------------------------------

        # HWC uint8 -> float32 [0,1]
        x = rgb.astype(np.float32) / 255.0

        # HWC -> CHW
        x = np.transpose(x, (2, 0, 1))  # [3,H,W]

        # tensor [1,3,H,W]
        x = torch.from_numpy(x).unsqueeze(0).to(self.device, non_blocking=True)

        # (선택) normalize (학습 시 사용했으므로 True 권장)
        if self.USE_IMAGENET_NORM:
            x = (x - self.imagenet_mean) / self.imagenet_std

        return x


def main(args=None):
    rclpy.init(args=args)
    node = MobileNetInferenceNode()
    rclpy.spin(node)
    node.destroy_node()

    # OpenCV 창 닫기
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()