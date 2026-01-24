#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage

# ✅ 너가 올린 모델 파일 기준
from valet_parking.mobilenetv3s_parking_model_pretrained import MultiCamParkingModel

# 학습 코드와 동일한 해상도 :contentReference[oaicite:4]{index=4}
IMG_WIDTH = 224
IMG_HEIGHT = 224


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def decode_jpeg_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img


def preprocess_like_train(img_bgr: np.ndarray) -> np.ndarray:
    """
    학습 코드와 동일하게 맞춤 :contentReference[oaicite:5]{index=5}
    - BGR -> RGB
    - resize (224,224)
    - (H,W,C) -> (C,H,W)
    - dtype float32 (0~255 유지)  ※ 모델이 내부에서 minus1_1 정규화 수행 :contentReference[oaicite:6]{index=6}
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.transpose(img, (2, 0, 1))  # (3,224,224)
    return img.astype(np.float32)


class PolicyInferNode(Node):
    def __init__(self):
        super().__init__('policy_infer_node')

        # ✅ 학습 코드 cam_cols와 동일 순서 :contentReference[oaicite:7]{index=7}
        # (front, rear, left, right)
        self.cams: List[str] = list(self.declare_parameter(
            'cams', ['front_cam', 'rear_cam', 'left_cam', 'right_cam']
        ).value)

        self.sync_slop = float(self.declare_parameter('sync_slop_sec', 0.10).value)
        self.pub_hz = float(self.declare_parameter('publish_hz', 10.0).value)

        # 모델 관련
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_path = str(self.declare_parameter('ckpt_path', '/home/sechankim/ros2_ws/src/valet_parking/valet_parking/best_model.pth').value).strip()

        self.out_max_linear = float(self.declare_parameter('out_max_linear', 0.25).value)
        self.out_max_angular = float(self.declare_parameter('out_max_angular', 2.0).value)

        # ✅ 모델 생성: 학습과 동일하게 pretrained=False, input_norm 기본 minus1_1 
        self.model = MultiCamParkingModel(pretrained=True).to(self.device).eval()

        if not self.ckpt_path:
            raise RuntimeError("ckpt_path is empty. Set ckpt_path to best_model.pth")

        sd = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)  # 학습은 state_dict 저장 :contentReference[oaicite:9]{index=9}
        self.get_logger().info(f'[POLICY] loaded ckpt: {self.ckpt_path} on {self.device}')

        # image buffer: cam -> (t, jpeg_bytes)
        self.img_buf: Dict[str, Tuple[float, bytes]] = {}

        # pub
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_policy', 10)

        # subs
        for cam in self.cams:
            topic = f'/{cam}/image/compressed'
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, c=cam: self.on_img(c, msg),
                1
            )

        # timer
        period = 1.0 / self.pub_hz if self.pub_hz > 0 else 0.1
        self.create_timer(period, self.tick)

        self.get_logger().info(
            f'[POLICY] ready cams={self.cams} sync_slop={self.sync_slop}s pub_hz={self.pub_hz}'
        )

    def on_img(self, cam: str, msg: CompressedImage):
        # cam 이름이 예상과 다르면 무시(실수 방지)
        if cam not in self.cams:
            return

        t = stamp_to_sec(msg.header.stamp)
        self.img_buf[cam] = (t, bytes(msg.data))

        # 간단 GC
        if len(self.img_buf) > 16:
            oldest_cam = min(self.img_buf.keys(), key=lambda k: self.img_buf[k][0])
            del self.img_buf[oldest_cam]

    def _pop_synced(self) -> Optional[Dict[str, bytes]]:
        # 4개 다 있어야 함
        for cam in self.cams:
            if cam not in self.img_buf:
                return None

        times = [self.img_buf[cam][0] for cam in self.cams]
        t_min, t_max = min(times), max(times)

        if (t_max - t_min) > self.sync_slop:
            # sync 실패면 oldest 버리고 다시 시도
            oldest_cam = min(self.cams, key=lambda c: self.img_buf[c][0])
            del self.img_buf[oldest_cam]
            return None

        out = {cam: self.img_buf[cam][1] for cam in self.cams}
        self.img_buf.clear()
        return out

    @torch.no_grad()
    def tick(self):
        synced = self._pop_synced()
        if synced is None:
            # 프레임 아직 안 맞으면 안전하게 0
            self.pub_cmd.publish(Twist())
            return

        # 4 cam decode + preprocess (학습 정합)
        images_4 = []
        for cam in self.cams:
            bgr = decode_jpeg_to_bgr(synced[cam])
            if bgr is None:
                self.pub_cmd.publish(Twist())
                return
            img_chw = preprocess_like_train(bgr)  # (3,224,224) float32 0~255
            images_4.append(img_chw)

        # (4,3,224,224) -> (1,4,3,224,224)
        x = np.stack(images_4, axis=0)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)  # float32

        # 모델 forward: out shape (B,2) :contentReference[oaicite:10]{index=10}
        y = self.model(x)[0]  # (2,)

        v = float(y[0].item())
        w = float(y[1].item())

        # 출력 clamp
        v = max(-self.out_max_linear, min(self.out_max_linear, v))
        w = max(-self.out_max_angular, min(self.out_max_angular, w))

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
