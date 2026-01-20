#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import CompressedImage
import cv2


class ParkingCompressedWebcamPublisher(Node):
    """
    - OpenCV로 /dev/front_cam, /dev/rear_cam 같은 "고정 symlink"를 직접 열어서
      front/rear 2개 카메라를 CompressedImage(jpeg)로 퍼블리시.
    - 토픽은 의미 기반으로 고정:
        /front/image/compressed
        /rear/image/compressed
    """

    def __init__(self):
        super().__init__('webcam_compressed_pub_parking')

        # ✅ index 대신 udev symlink 경로 사용
        self.declare_parameter('front_dev', '/dev/front_cam')
        self.declare_parameter('rear_dev',  '/dev/rear_cam')

        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 10.0)

        self.declare_parameter('jpeg_quality', 50)
        self.declare_parameter('use_mjpeg_fourcc', True)
        self.declare_parameter('buffer_size', 1)
        self.declare_parameter('period_sec', 0.1)

        self.front_dev = str(self.get_parameter('front_dev').value)
        self.rear_dev  = str(self.get_parameter('rear_dev').value)

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)

        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.use_mjpeg_fourcc = bool(self.get_parameter('use_mjpeg_fourcc').value)
        self.buffer_size = int(self.get_parameter('buffer_size').value)
        self.period_sec = float(self.get_parameter('period_sec').value)

        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        self.cb_group = ReentrantCallbackGroup()

        # publishers
        self.pub_front = self.create_publisher(CompressedImage, '/front/image/compressed', 1)
        self.pub_rear  = self.create_publisher(CompressedImage, '/rear/image/compressed', 1)

        # cameras
        self.cap_front = self._open_cam(self.front_dev, "front")
        self.cap_rear  = self._open_cam(self.rear_dev,  "rear")

        # timers
        self.timer_front = self.create_timer(
            self.period_sec,
            lambda: self._cam_cb(self.cap_front, self.pub_front, "front"),
            callback_group=self.cb_group
        )
        self.timer_rear = self.create_timer(
            self.period_sec,
            lambda: self._cam_cb(self.cap_rear, self.pub_rear, "rear"),
            callback_group=self.cb_group
        )

        self.get_logger().info(
            f"[READY] front_dev={self.front_dev}, rear_dev={self.rear_dev} | "
            f"{self.width}x{self.height}@{self.fps} | jpeg_quality={self.jpeg_quality}"
        )

    def _open_cam(self, dev: str, tag: str):
        # ✅ 팀원이 말한 것처럼 CAP_V4L2 명시 권장
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if not cap.isOpened():
            self.get_logger().error(f"[{tag}] Cannot open camera dev={dev}")
            return cap

        if self.use_mjpeg_fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self.get_logger().info(f"[{tag}] Opened camera dev={dev}")
        return cap

    def _cam_cb(self, cap, pub, frame_id: str):
        if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            return

        ok, enc = cv2.imencode('.jpg', frame, self.encode_param)
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        msg.data = enc.tobytes()
        pub.publish(msg)

    def close(self):
        for cap in [self.cap_front, self.cap_rear]:
            try:
                if cap is not None and cap.isOpened():
                    cap.release()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = ParkingCompressedWebcamPublisher()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
