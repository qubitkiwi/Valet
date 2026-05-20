#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, String  # ✅ String 추가
import json # ✅ JSON 파싱용

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# -------------------------------------------------
# 1. Model Architecture (Must match training code)
# -------------------------------------------------
class RepViTMultiHead(nn.Module):
    def __init__(self, model_name='repvit_m0_9', num_classes=4):
        super(RepViTMultiHead, self).__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        except Exception:
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            num_features = features.shape[1]
            
        self.reg_head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) 
        )
        
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            
        raw_reg = self.reg_head(features)
        reg_out = torch.tanh(raw_reg)
        reg_out = reg_out * 2.0
        
        cls_out = self.cls_head(features)
        return reg_out, cls_out

# -------------------------------------------------
# 2. ROS2 Inference Node
# -------------------------------------------------
class DrivingInferenceNode(Node):
    def __init__(self):
        super().__init__('driving_inference_node')
        
        self.declare_parameter('model_path', './repvit_final_v2_best.pth')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('robot_status_topic', '/robot_status') # ✅ 파라미터화

        model_path = self.get_parameter('model_path').value
        device_name = self.get_parameter('device').value
        robot_status_topic = self.get_parameter('robot_status_topic').value
        
        self.device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
        
        # ✅ 현재 로봇 상태 변수 초기화
        self.current_robot_status = "unknown"

        self.get_logger().info(f"🔄 Loading Model from {model_path} to {self.device}...")

        try:
            self.model = RepViTMultiHead(model_name='repvit_m0_9', num_classes=4)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("✅ Model Loaded Successfully!")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")
            raise e

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Subscriptions
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/front_cam/image/compressed',
            self.image_callback,
            1
        )

        # ✅ Robot Status Subscriber 추가
        self.status_sub = self.create_subscription(
            String,
            robot_status_topic,
            self.status_callback,
            10
        )

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/driving/raw_cmd', 10)
        self.pub_sign = self.create_publisher(Int32, '/sign_class', 10)
        
        self.get_logger().info(f"🚀 Inference Node Started. (Waiting for status: driving/call)")

    # ✅ Robot Status 콜백 함수
    def status_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.current_robot_status = data.get("mode", "unknown")
        except Exception as e:
            self.get_logger().error(f"Status parsing error: {e}")

    def image_callback(self, msg):
        # ✅ [조건 추가] driving 또는 call 상태가 아니면 추론 및 발행을 하지 않음
        if self.current_robot_status not in ["driving", "call"]:
            return

        try:
            # 1. CompressedImage -> OpenCV Image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                return

            # 2. OpenCV(BGR) -> PIL(RGB)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)

            # 3. Preprocessing
            input_tensor = self.transform(pil_image)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            # 4. Inference
            with torch.no_grad():
                pred_cmd, pred_cls = self.model(input_tensor)

            # 5. Post-processing
            cmd_vel_np = pred_cmd.cpu().numpy()[0]
            linear_x = float(cmd_vel_np[0])
            angular_z = float(cmd_vel_np[1])

            # Classification Result
            _, cls_idx = torch.max(pred_cls, 1)
            sign_class = int(cls_idx.item())

            # Logic Modification: Class에 따른 Angular Gain 조절
            if sign_class == 1 or sign_class == 3:
                angular_z *= 0.5
            elif sign_class == 2:
                angular_z *= 1.5

            # 6. Publish Messages
            twist_msg = Twist()
            twist_msg.linear.x = linear_x
            twist_msg.angular.z = angular_z
            self.pub_cmd.publish(twist_msg)

            sign_msg = Int32()
            sign_msg.data = sign_class
            self.pub_sign.publish(sign_msg)

            self.get_logger().info(f"[{self.current_robot_status}] Lin: {linear_x:.2f}, Ang: {angular_z:.2f}, Class: {sign_class}")

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DrivingInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()