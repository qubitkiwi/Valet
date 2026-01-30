# -*- coding: utf-8 -*-

import os
import sys
import time
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image, Joy
from cv_bridge import CvBridge
from std_msgs.msg import String


# ==========================================
# 1. Import Models
# ==========================================
# (A) Sign/YOLO용 모델
from .deeplabv3_sign import DeepLabV3MultiTask
# (B) MultiCam용 모델
from valet_parking.mobilenetv3s_parking_model_pretrained import MultiCamParkingModel


# ==========================================
# 2. Helpers & Constants
# ==========================================
IMG_WIDTH = 224
IMG_HEIGHT = 224

CROP_SETTINGS = {
    'front_cam': (0, 480, 0, 640),
    'rear_cam':  (0, 480, 0, 640),
    'left_cam':  (0, 300, 100, 640),
    'right_cam': (00, 350, 0, 540)
}

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def decode_jpeg_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def preprocess_multicam(img_bgr: np.ndarray, cam_name: str) -> np.ndarray:
    """Mode B (MultiCam) 전용 전처리"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Crop 적용
    if cam_name in CROP_SETTINGS:
        y1, y2, x1, x2 = CROP_SETTINGS[cam_name]
        h, w, _ = img.shape
        y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
        
        if y2 > y1 and x2 > x1:
            img = img[y1:y2, x1:x2]
        else:
            pass

    # Resize
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.transpose(img, (2, 0, 1)) # (3,224,224)
    return img.astype(np.float32)


class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        
        # =================================================
        # [State] 모드 관리 (0: Sign/YOLO, 1: MultiCam)
        # =================================================
        self.is_running = False
        self.mode = 0  # 기본값: Sign/YOLO 모드
        self.mode_names = {0: "MODE_A_SIGN_YOLO", 1: "MODE_B_MULTICAM"}
        self.prev_joy_btn = 0 # 버튼 디바운스용

        # =================================================
        # [Params] Mode A (DeepLab + YOLO)
        # =================================================
        self.declare_parameter('MODEL_PATH_A', 'best_model_with_sign2.pth') # DeepLab
        self.declare_parameter('YOLO_PATH', 'yolo11n.pt')
        self.declare_parameter('LINEAR_GAIN', 0.8)
        self.declare_parameter('STEERING_GAIN', 1.0)
        
        self.A_ckpt = self.get_parameter('MODEL_PATH_A').value
        self.A_yolo = self.get_parameter('YOLO_PATH').value
        self.A_lin_gain = self.get_parameter('LINEAR_GAIN').value
        self.A_ang_gain = self.get_parameter('STEERING_GAIN').value

        # YOLO Voting Params (축약)
        self.STOP_CLASSES = [0, 2] # Person, Car
        self.STOP_AREA_RATIO = 0.10
        self.STOP_HARD_AREA_RATIO = 0.18
        self.VOTE_K = 2
        
        # State for A
        self._stop_latched = False
        self._stop_count = 0
        self._release_count = 0
        self._area_ema = None
        self._area_ema_prev = None

        # =================================================
        # [Params] Mode B (MultiCam)
        # =================================================
        self.declare_parameter('MODEL_PATH_B', '/home/hyunii/test_ws/mobilenetv3s_pretrained_up_mid_crop_LR_onecycle_batch256_epoch100_lr0001.pth')
        self.B_ckpt = self.get_parameter('MODEL_PATH_B').value
        self.B_out_lin = 0.25
        self.B_out_ang = 2.0
        self.cams = ['front_cam', 'rear_cam', 'left_cam', 'right_cam']
        self.sync_slop = 0.1
        
        # Buffer for B
        self.img_buf: Dict[str, Tuple[float, bytes]] = {}

        # =================================================
        # [Load Models]
        # =================================================
        self.get_logger().info("--- Loading Models ---")
        
        # 1. Load A (DeepLab)
        self.model_deeplab, self.ctrl_mean, self.ctrl_std, _ = self._load_deeplab(self.A_ckpt)
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 2. Load A (YOLO)
        self.model_yolo = YOLO(self.A_yolo)
        
        # 3. Load B (MultiCam)
        self.model_multicam = MultiCamParkingModel(pretrained=True).to(self.device).eval()
        try:
            sd = torch.load(self.B_ckpt, map_location=self.device)
            self.model_multicam.load_state_dict(sd, strict=True)
            self.get_logger().info(f"[Mode B] Loaded MultiCam: {self.B_ckpt}")
        except Exception as e:
            self.get_logger().error(f"[Mode B] Load Failed: {e}")

        # =================================================
        # [Comm] Subs & Pubs
        # =================================================
        # Pub
        self.pub_cmd = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.pub_vis = self.create_publisher(Image, '/hybrid/vis', 10)
        

        # Joy Sub
        self.create_subscription(String, '/mode', self.on_mode, 10)
        self.create_subscription(String, '/joy/event', self.on_joy_event, 10)

        # Camera Subs (4개 모두 구독)
        for cam in self.cams:
            self.create_subscription(
                CompressedImage, 
                f'/{cam}/image/compressed', 
                lambda msg, c=cam: self.on_img(c, msg), 
                1
            )

        # Timer for Mode B (10Hz)
        self.create_timer(0.1, self.tick_mode_b)

        self.get_logger().info(f"✅ Hybrid Node Ready! Start Mode: {self.mode_names[self.mode]}")

    # =================================================
    # [추가] Joy Event Callback (START/END)
    # =================================================
    def on_joy_event(self, msg: String):
        try:
            data = json.loads(msg.data)
            evt_type = data.get("type", "")
            
            if evt_type == "START":
                if not self.is_running:
                    self.is_running = True
                    self.get_logger().info("🟢 [START] Autonomous Driving STARTED")
            
            elif evt_type == "END":
                if self.is_running:
                    self.is_running = False
                    self.pub_cmd.publish(Twist()) # 정지 명령 즉시 전송
                    self.get_logger().info("🔴 [END] Autonomous Driving STOPPED")

        except Exception as e:
            self.get_logger().error(f"Joy event parse error: {e}")

    # =================================================
    # Model Loader Helpers
    # =================================================
    def _load_deeplab(self, path):
        self.get_logger().info(f"[Mode A] Loading DeepLab: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False) # weights_only=False 주의
        
        # 데이터 구조 확인
        if "model" in ckpt:
             state_dict = ckpt["model"]
             num_signs = ckpt.get("num_signs", 2) # 기본값 안전장치
             c_mean = torch.tensor(ckpt["ctrl_mean"], device=self.device).view(1,2)
             c_std = torch.tensor(ckpt["ctrl_std"], device=self.device).view(1,2)
        else:
            # 구조가 다르면 예외처리 혹은 직접 로드
            raise RuntimeError("DeepLab Checkpoint format unknown")

        model = DeepLabV3MultiTask(num_signs=num_signs, pretrained=True).to(self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, c_mean, c_std, num_signs

    # =================================================
    # Callbacks
    # =================================================
    def on_mode(self, msg: String):
        m = (msg.data or '').strip().lower()
        if m == 'drive':
            new_mode = 0
        elif m == 'park':
            new_mode = 1
        else:
            return
    
        if new_mode != self.mode:
            self.mode = new_mode
            self.get_logger().warn(f"🔀 SWITCHED TO: {self.mode_names[self.mode]}")
            self.pub_cmd.publish(Twist())  # 안전 정지

    def on_img(self, cam_name, msg):
        # 1. 모든 카메라는 Buffer에 저장 (Mode B용)
        t = stamp_to_sec(msg.header.stamp)
        self.img_buf[cam_name] = (t, bytes(msg.data))
        
        # Buffer GC
        if len(self.img_buf) > 16:
            oldest_cam = min(self.img_buf.keys(), key=lambda k: self.img_buf[k][0])
            del self.img_buf[oldest_cam]

        # [추가] 정지 상태면 추론 안함
        if not self.is_running:
            return
        
        # 2. 만약 Mode A이고, 들어온게 'front_cam'이면 즉시 실행
        if self.mode == 0 and cam_name == 'front_cam':
            self.step_mode_a(msg)

    def tick_mode_b(self):
        # 정지 상태면 추론 안함
        if not self.is_running:
            return
        
        # Mode B일 때만 Timer로 동작
        if self.mode == 1:
            self.step_mode_b()

    # =================================================
    # Logic: Mode A (Sign + YOLO)
    # =================================================
    def step_mode_a(self, msg):
        # Decode
        img_bgr = decode_jpeg_to_bgr(msg.data)
        if img_bgr is None: return

        # 1. DeepLab Inference
        v_raw, w_raw, vis_base = self._infer_deeplab(img_bgr)
        
        # 2. YOLO & Voting
        stop_now, reason = self._infer_yolo_vote(img_bgr, vis_base)
        
        # 3. Control
        if stop_now:
            v_final = 0.0
        else:
            v_final = v_raw * self.A_lin_gain
        
        w_final = w_raw * self.A_ang_gain
        
        # Publish
        cmd = Twist()
        cmd.linear.x = float(np.clip(v_final, -0.5, 0.5))
        cmd.angular.z = float(np.clip(w_final, -3.0, 3.0))
        self.pub_cmd.publish(cmd)

        # Vis
        if vis_base is not None:
            cv2.putText(vis_base, f"MODE A: {reason}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_base, encoding='bgr8')
            self.pub_vis.publish(vis_msg)

    def _infer_deeplab(self, img_bgr):
        # Preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = (x - self.norm_mean) / self.norm_std
        x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model_deeplab(x)
        
        # Denormalize
        ctrl = out["control"] * self.ctrl_std + self.ctrl_mean
        v = float(ctrl[0, 0].item())
        w = float(ctrl[0, 1].item())
        
        return v, w, img_bgr.copy() # Vis용 복사

    def _infer_yolo_vote(self, img_bgr, vis_img):
        # YOLO
        results = self.model_yolo(img_bgr, conf=0.5, verbose=False)[0]
        
        found_stop = False
        max_area = 0.0
        
        h, w = img_bgr.shape[:2]
        total = h*w

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.STOP_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2-x1)*(y2-y1)/total
                
                # Vis
                cv2.rectangle(vis_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 2)
                
                if area > self.STOP_AREA_RATIO:
                    found_stop = True
                    max_area = max(max_area, area)

        # Latch Logic
        if found_stop:
            self._stop_count += 1
            self._release_count = 0
        else:
            self._stop_count = 0
            self._release_count += 1
            
        if not self._stop_latched and self._stop_count >= 3:
            self._stop_latched = True
        if self._stop_latched and self._release_count >= 5:
            self._stop_latched = False
            
        return self._stop_latched, f"Stop:{self._stop_latched}({max_area:.2f})"


    # =================================================
    # Logic: Mode B (MultiCam)
    # =================================================
    def step_mode_b(self):
        # Sync Logic
        # _pop_synced랑 동일
        cams_needed = ['front_cam', 'rear_cam', 'left_cam', 'right_cam']
        if not all(c in self.img_buf for c in cams_needed):
            return # not ready
            
        times = [self.img_buf[c][0] for c in cams_needed]
        if max(times) - min(times) > self.sync_slop:
            # Sync fail -> drop oldest
            oldest = min(cams_needed, key=lambda c: self.img_buf[c][0])
            del self.img_buf[oldest]
            return

        # Prepare Batch
        imgs = []
        for c in cams_needed:
            bgr = decode_jpeg_to_bgr(self.img_buf[c][1])
            if bgr is None: return
            imgs.append(preprocess_multicam(bgr, c))
        
        # Clear used
        self.img_buf.clear()

        # Inference
        x = np.stack(imgs, axis=0) # (4, 3, 224, 224)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device) # (1, 4, 3, 224, 224)
        
        with torch.no_grad():
            y = self.model_multicam(x)[0]
        
        v = float(y[0].item())
        w = float(y[1].item())
        
        v = max(-self.B_out_lin, min(self.B_out_lin, v))
        w = max(-self.B_out_ang, min(self.B_out_ang, w))
        
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)
        
        # (Optional) Print status
        # print(f"Mode B: v={v:.3f}, w={w:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()