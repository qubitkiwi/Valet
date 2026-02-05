#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

import cv2
import torch
import numpy as np


from ultralytics import YOLO

# ✅ 네가 학습에 쓴 모델
from deeplabv3_sign import DeepLabV3MultiTask

print("=== RUNNING FILE:", __file__)


class DeepLabSignYoloInferenceNode(Node):
    def __init__(self):
        super().__init__('deeplab_sign_yolo_inference_node')

        # -------------------------
        # DeepLab Params (주행용)
        # -------------------------
        # self.declare_parameter('CROP_HEIGHT', 100)
        self.declare_parameter('LINEAR_GAIN', 0.8)
        self.declare_parameter('STEERING_GAIN', 1.0)

        # ✅ 멀티태스크 best ckpt
        self.declare_parameter('MODEL_PATH', 'best_model_with_sign2.pth')

        # -------------------------
        # YOLO Params (장애물 감지용)
        # -------------------------
        self.declare_parameter('YOLO_PATH', 'yolo11n.pt')
        self.declare_parameter('YOLO_CONF', 0.5)
        self.declare_parameter('STOP_CLASSES', [0, 2])       # 0=person, 2=car (COCO)
        self.declare_parameter('STOP_AREA_RATIO', 0.10)      # vote용
        self.declare_parameter('STOP_HARD_AREA_RATIO', 0.18) # hard-stop

        # OpenCV heuristic (bottom y)
        self.declare_parameter('STOP_BOTTOM_RATIO', 0.80)

        # TTC-like (area ratio EMA growth)
        self.declare_parameter('TTC_AREA_EMA_ALPHA', 0.30)
        self.declare_parameter('TTC_AREA_GROWTH_TH', 0.015)

        # Voting / debounce
        self.declare_parameter('VOTE_K', 2)
        self.declare_parameter('STOP_DEBOUNCE_FRAMES', 3)
        self.declare_parameter('RELEASE_DEBOUNCE_FRAMES', 5)

        # -------------------------
        # Common Params
        # -------------------------
        self.declare_parameter('input_video', '/front_cam/image/compressed')
        self.declare_parameter('output_cmd_topic', '/controller/cmd_vel')
        # self.declare_parameter('OUT_W', 320)
        # self.declare_parameter('OUT_H', 192)

        # Safety Clamps
        self.declare_parameter('CLAMP_LINEAR', True)
        self.declare_parameter('LINEAR_MIN', -0.5)
        self.declare_parameter('LINEAR_MAX',  0.5)
        self.declare_parameter('ANGULAR_MIN', -3.0)
        self.declare_parameter('ANGULAR_MAX',  3.0)

        self.declare_parameter('LOG_EVERY_N', 10)

        # Visualization publish
        self.declare_parameter('PUB_VIS', True)
        self.declare_parameter('vis_topic', '/deeplab_sign_yolo/vis')
        self.declare_parameter('DRAW_SIGN', True)  # sign 예측도 오버레이로 그릴지

        # -------------------------
        # Read Params
        # -------------------------
        # self.CROP_HEIGHT = int(self.get_parameter('CROP_HEIGHT').value)
        self.LINEAR_GAIN = float(self.get_parameter('LINEAR_GAIN').value)
        self.STEERING_GAIN = float(self.get_parameter('STEERING_GAIN').value)
        self.MODEL_PATH = str(self.get_parameter('MODEL_PATH').value)

        self.YOLO_PATH = str(self.get_parameter('YOLO_PATH').value)
        self.YOLO_CONF = float(self.get_parameter('YOLO_CONF').value)
        self.STOP_CLASSES = list(self.get_parameter('STOP_CLASSES').value)
        self.STOP_AREA_RATIO = float(self.get_parameter('STOP_AREA_RATIO').value)
        self.STOP_HARD_AREA_RATIO = float(self.get_parameter('STOP_HARD_AREA_RATIO').value)

        self.STOP_BOTTOM_RATIO = float(self.get_parameter('STOP_BOTTOM_RATIO').value)
        self.TTC_AREA_EMA_ALPHA = float(self.get_parameter('TTC_AREA_EMA_ALPHA').value)
        self.TTC_AREA_GROWTH_TH = float(self.get_parameter('TTC_AREA_GROWTH_TH').value)

        self.VOTE_K = int(self.get_parameter('VOTE_K').value)
        self.STOP_DEBOUNCE_FRAMES = int(self.get_parameter('STOP_DEBOUNCE_FRAMES').value)
        self.RELEASE_DEBOUNCE_FRAMES = int(self.get_parameter('RELEASE_DEBOUNCE_FRAMES').value)

        self.input_video = str(self.get_parameter('input_video').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        # self.OUT_W = int(self.get_parameter('OUT_W').value)
        # self.OUT_H = int(self.get_parameter('OUT_H').value)

        self.CLAMP_LINEAR = bool(self.get_parameter('CLAMP_LINEAR').value)
        self.LINEAR_MIN = float(self.get_parameter('LINEAR_MIN').value)
        self.LINEAR_MAX = float(self.get_parameter('LINEAR_MAX').value)
        self.ANGULAR_MIN = float(self.get_parameter('ANGULAR_MIN').value)
        self.ANGULAR_MAX = float(self.get_parameter('ANGULAR_MAX').value)

        self.LOG_EVERY_N = int(self.get_parameter('LOG_EVERY_N').value)

        self.PUB_VIS = bool(self.get_parameter('PUB_VIS').value)
        self.vis_topic = str(self.get_parameter('vis_topic').value)
        self.DRAW_SIGN = bool(self.get_parameter('DRAW_SIGN').value)

        self._frame_count = 0

        # -------------------------
        # Voting State
        # -------------------------
        self._stop_latched = False
        self._stop_count = 0
        self._release_count = 0

        # TTC(면적비 변화) EMA state
        self._area_ema = None
        self._area_ema_prev = None

        # -------------------------
        # ROS Setup
        # -------------------------
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.input_video,
            self.image_callback,
            1
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.output_cmd_topic,
            10
        )

        self.vis_pub = self.create_publisher(
            Image,
            self.vis_topic,
            10
        )

        # -------------------------
        # Load Models
        # -------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) DeepLab MultiTask (Driving + Sign)
        self.model, self.ctrl_mean_t, self.ctrl_std_t, self.num_signs = self.load_multitask_model(
            self.MODEL_PATH, self.device
        )

        # ImageNet normalize (torchvision pretrained)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 2) YOLO
        self.get_logger().info(f"Loading YOLO model from {self.YOLO_PATH}...")
        self.yolo_model = YOLO(self.YOLO_PATH)

        self.get_logger().info(
            f"✅ Node Ready! | input={self.input_video} | cmd_pub={self.output_cmd_topic} | vis_pub={self.vis_topic}"
        )

    # -------------------------
    # Model Loader (멀티태스크 체크포인트 전용)
    # -------------------------
    def load_multitask_model(self, ckpt_path: str, device: torch.device):
        self.get_logger().info(f"Loading MultiTask Model from {ckpt_path}...")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError("Checkpoint must be a dict containing key 'model' (your training save format).")

        num_signs = int(ckpt.get("num_signs", 0))
        if num_signs <= 0:
            raise RuntimeError("Checkpoint missing valid 'num_signs'.")

        ctrl_mean = ckpt.get("ctrl_mean", None)
        ctrl_std  = ckpt.get("ctrl_std", None)
        if ctrl_mean is None or ctrl_std is None:
            raise RuntimeError("Checkpoint missing 'ctrl_mean'/'ctrl_std' needed for denormalization.")

        ctrl_mean_t = torch.tensor(ctrl_mean, dtype=torch.float32, device=device).view(1, 2)
        ctrl_std_t  = torch.tensor(ctrl_std,  dtype=torch.float32, device=device).view(1, 2)

        model = DeepLabV3MultiTask(num_signs=num_signs, pretrained=True).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        self.get_logger().info(f"Loaded: num_signs={num_signs}, ctrl_mean={ctrl_mean}, ctrl_std={ctrl_std}")
        return model, ctrl_mean_t, ctrl_std_t, num_signs

    # -------------------------
    # Main Callback
    # -------------------------
    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return

        # 1) Driving Inference (DeepLab multitask)
        try:
            v_base, w_base, sign_id, sign_conf, vis_base = self.run_multitask_inference(image_bgr)
        except Exception as e:
            self.get_logger().error(f"MultiTask Inference Error: {e}")
            return

        v_final = v_base * self.LINEAR_GAIN
        w_final = w_base * self.STEERING_GAIN

        # 2) YOLO + Voting Decision
        stop_now = False
        reason = "none"
        vote_cnt = 0
        try:
            yolo_feat, vis_img = self.run_yolo_inference(image_bgr, vis_base)
            stop_now, reason, vote_cnt = self.decide_stop_by_voting(yolo_feat)

            if stop_now:
                v_final = 0.0
        except Exception as e:
            self.get_logger().error(f"YOLO/Voting Error: {e}")
            vis_img = vis_base

        # 3) Safety Clamp
        if self.CLAMP_LINEAR:
            v_final = float(np.clip(v_final, self.LINEAR_MIN, self.LINEAR_MAX))
        w_final = float(np.clip(w_final, self.ANGULAR_MIN, self.ANGULAR_MAX))

        # 4) Publish Command
        twist = Twist()
        twist.linear.x = float(v_final)
        twist.angular.z = float(w_final)
        self.cmd_pub.publish(twist)

        # 5) Publish Visualization
        if self.PUB_VIS and vis_img is not None:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header = msg.header
            self.vis_pub.publish(vis_msg)

        # Log
        self._frame_count += 1
        if self.LOG_EVERY_N > 0 and (self._frame_count % self.LOG_EVERY_N == 0):
            status = "STOP" if self._stop_latched else "GO"
            self.get_logger().info(
                f"[{status}] v={v_final:.3f}, w={w_final:.3f} | base_v={v_base:.3f} "
                f"| sign={sign_id}({sign_conf:.2f}) | votes={vote_cnt} | {reason}"
            )

    # -------------------------
    # Logic: DeepLab MultiTask (Driving + Sign)
    # -------------------------
    def run_multitask_inference(self, image_bgr):
        """
        returns:
          v, w: denormalized control outputs (float)
          sign_id, sign_conf
          vis_base: visualization base image (OUT_W x OUT_H BGR)
        """
        # crop & resize
        # img_crop = image_bgr[self.CROP_HEIGHT:480, :]
        img_crop = image_bgr
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        # img_rgb = cv2.resize(img_rgb, (self.OUT_W, self.OUT_H), interpolation=cv2.INTER_LINEAR)
        x = img_rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)  # {"control": [1,2] (normalized), "signs": [1,C] logits}

        ctrl_n = out["control"]  # normalized
        logits = out["signs"]

        # ✅ denormalize control
        ctrl = ctrl_n * self.ctrl_std_t + self.ctrl_mean_t
        v = float(ctrl[0, 0].item())
        w = float(ctrl[0, 1].item())

        # sign prediction
        prob = torch.softmax(logits, dim=1)
        conf, pred = torch.max(prob, dim=1)
        sign_id = int(pred.item())
        sign_conf = float(conf.item())

        # vis base (BGR)
        vis_base = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.DRAW_SIGN:
            cv2.putText(
                vis_base,
                f"SIGN={sign_id} ({sign_conf:.3f}) | v={v:.6f} w={w:.6f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return v, w, sign_id, sign_conf, vis_base

    # -------------------------
    # Logic: YOLO (Obstacle Feature Extraction)
    # -------------------------
    def run_yolo_inference(self, image_bgr, vis_base):
        """
        returns:
          - best_feat: dict with {found, cls_id, conf, area_ratio, bottom_ratio, box_xyxy}
          - vis_img: visualization image (OUT_W x OUT_H)
        """
        results = self.yolo_model(image_bgr, conf=self.YOLO_CONF, verbose=False)
        result = results[0]

        img_h, img_w = image_bgr.shape[:2]
        total_area = float(img_h * img_w)

        # if vis_base is None:
        #     vis_img = cv2.resize(image_bgr, (self.OUT_W, self.OUT_H))
        # else:
        #     vis_img = vis_base.copy()

        vis_img = vis_base.copy() if vis_base is not None else image_bgr.copy()

        vis_h, vis_w = vis_img.shape[:2]
        scale_x = vis_w / img_w
        scale_y = vis_h / img_h

        best = {
            "found": False,
            "cls_id": None,
            "conf": 0.0,
            "area_ratio": 0.0,
            "bottom_ratio": 0.0,
            "box_xyxy": None,
        }

        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in self.STOP_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = float((x2 - x1) * (y2 - y1))
            area_ratio = float(box_area / total_area)
            bottom_ratio = float(y2 / img_h)

            # draw on vis
            vx1, vy1 = int(x1 * scale_x), int(y1 * scale_y)
            vx2, vy2 = int(x2 * scale_x), int(y2 * scale_y)

            label = f"id{cls_id} a={area_ratio:.2f} b={bottom_ratio:.2f}"
            cv2.rectangle(vis_img, (vx1, vy1), (vx2, vy2), (0, 255, 255), 2)
            cv2.putText(vis_img, label, (vx1, max(0, vy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # best update by area_ratio
            if (not best["found"]) or (area_ratio > best["area_ratio"]):
                best.update({
                    "found": True,
                    "cls_id": cls_id,
                    "conf": conf,
                    "area_ratio": area_ratio,
                    "bottom_ratio": bottom_ratio,
                    "box_xyxy": (x1, y1, x2, y2),
                })

        return best, vis_img

    # -------------------------
    # Voting Decision
    # -------------------------
    def decide_stop_by_voting(self, yolo_feat: dict):
        votes = 0
        reasons = []

        if not yolo_feat.get("found", False):
            stop_now = self._update_latch(False)
            return stop_now, "no_yolo", 0

        area_ratio = float(yolo_feat["area_ratio"])
        bottom_ratio = float(yolo_feat["bottom_ratio"])

        # Hard stop
        if area_ratio >= self.STOP_HARD_AREA_RATIO:
            stop_now = self._update_latch(True, hard=True)
            return stop_now, f"hard_area {area_ratio:.3f}", 999

        # Vote A: area ratio
        if area_ratio >= self.STOP_AREA_RATIO:
            votes += 1
            reasons.append(f"area {area_ratio:.3f}")

        # Vote B: bottom ratio
        if bottom_ratio >= self.STOP_BOTTOM_RATIO:
            votes += 1
            reasons.append(f"bottom {bottom_ratio:.3f}")

        # Vote C: TTC-like EMA growth
        if self._area_ema is None:
            self._area_ema = area_ratio
            self._area_ema_prev = area_ratio
        else:
            self._area_ema_prev = self._area_ema
            a = self.TTC_AREA_EMA_ALPHA
            self._area_ema = (1 - a) * self._area_ema + a * area_ratio

        growth = float(self._area_ema - (self._area_ema_prev if self._area_ema_prev is not None else self._area_ema))
        if growth >= self.TTC_AREA_GROWTH_TH:
            votes += 1
            reasons.append(f"ttc_grow {growth:.4f}")

        stop_vote = (votes >= self.VOTE_K)
        stop_now = self._update_latch(stop_vote)

        reason = ",".join(reasons) if reasons else "no_vote"
        return stop_now, reason, votes

    def _update_latch(self, stop_vote: bool, hard: bool = False):
        if hard:
            self._stop_latched = True
            self._stop_count = 0
            self._release_count = 0
            return True

        if stop_vote:
            self._stop_count += 1
            self._release_count = 0
        else:
            self._release_count += 1
            self._stop_count = 0

        if (not self._stop_latched) and (self._stop_count >= self.STOP_DEBOUNCE_FRAMES):
            self._stop_latched = True

        if self._stop_latched and (self._release_count >= self.RELEASE_DEBOUNCE_FRAMES):
            self._stop_latched = False

        return self._stop_latched


def main(args=None):
    print("=== ENTER main() ===")
    rclpy.init(args=args)
    node = DeepLabSignYoloInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
