#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class CollectorNode(Node):
    def __init__(self):
        super().__init__('collector_node')

        self.root = Path(str(self.declare_parameter(
            'dataset_root',
            str(Path('~/ros2_ws/src/dataset/valet_parking').expanduser())
        ).value)).expanduser()

        self.cams = list(self.declare_parameter(
            'cams', ['left_cam', 'right_cam', 'rear_cam', 'front_cam']
        ).value)

        self.sync_slop = float(self.declare_parameter('sync_slop_sec', 0.1).value)
        self.v_th = float(self.declare_parameter('moving_v_th', 0.02).value)
        self.w_th = float(self.declare_parameter('moving_w_th', 0.05).value)

        self.root.mkdir(parents=True, exist_ok=True)

        self.active = False
        self.record_armed = False
        self.frame_idx = 0

        self.run_name: Optional[str] = None
        self.episode_name: Optional[str] = None
        self.ep_dir: Optional[Path] = None
        self.csv_f = None
        self.csv_wr = None

        self.last_cmd: Optional[Tuple[float, float]] = None  # (v,w)
        self.img_buf: Dict[str, Tuple[float, bytes]] = {}

        self.create_subscription(String, '/valet/event', self.on_event, 10)
        self.create_subscription(Bool, '/record_control', self.on_record_control, 10)
        self.create_subscription(Twist, '/controller/cmd_vel', self.on_cmd, 10)

        for cam in self.cams:
            topic = f'/{cam}/image/compressed'
            self.create_subscription(CompressedImage, topic, lambda msg, c=cam: self.on_img(c, msg), 1)

        self.get_logger().info(f'[COLLECTOR] ready root={self.root} cams={self.cams}')

    def moving(self) -> bool:
        if self.last_cmd is None:
            return False
        v, w = self.last_cmd
        return (abs(v) > self.v_th) or (abs(w) > self.w_th)

    def _open_episode(self, run_name: str, episode_name: str):
        self.run_name = run_name
        self.episode_name = episode_name
        self.ep_dir = self.root / run_name / episode_name

        for cam in self.cams:
            (self.ep_dir / cam).mkdir(parents=True, exist_ok=True)

        self.csv_f = open(self.ep_dir / 'actions.csv', 'w', newline='')
        self.csv_wr = csv.writer(self.csv_f)

        # ✅ 헤더: idx,t_sec 제거 (너 요청)
        self.csv_wr.writerow([
            'front_cam', 'rear_cam', 'left_cam', 'right_cam',
            'linear_x', 'angular_z'
        ])

        self.active = True
        self.frame_idx = 0
        self.img_buf.clear()

        self.get_logger().info(f'[COLLECTOR] EP_START dir={self.ep_dir}')

    def _close_episode(self, delete_if_empty=False):
        if not self.active:
            return

        try:
            if self.csv_f:
                self.csv_f.close()
        except Exception:
            pass

        if delete_if_empty and self.frame_idx == 0 and self.ep_dir is not None:
            try:
                shutil.rmtree(self.ep_dir)
                self.get_logger().warn('[COLLECTOR] empty episode deleted')
            except Exception as e:
                self.get_logger().warn(f'[COLLECTOR] delete failed: {e}')

        self.active = False
        self.frame_idx = 0
        self.run_name = None
        self.episode_name = None
        self.ep_dir = None
        self.csv_f = None
        self.csv_wr = None
        self.img_buf.clear()

        self.get_logger().info('[COLLECTOR] EP_END closed')

    def _try_flush_frame(self):
        if not self.active:
            return
        if not self.record_armed:
            return
        if not self.moving():
            return
        if self.ep_dir is None or self.csv_wr is None:
            return

        for cam in self.cams:
            if cam not in self.img_buf:
                return

        times = [self.img_buf[cam][0] for cam in self.cams]
        t_min, t_max = min(times), max(times)

        if (t_max - t_min) > self.sync_slop:
            oldest_cam = min(self.cams, key=lambda c: self.img_buf[c][0])
            del self.img_buf[oldest_cam]
            return

        # 상대 경로
        rel = {cam: f'{cam}/{self.frame_idx:06d}.jpg' for cam in self.cams}

        # 저장
        for cam in self.cams:
            _, data = self.img_buf[cam]
            out_path = self.ep_dir / rel[cam]
            with open(out_path, 'wb') as f:
                f.write(data)

        v, w = self.last_cmd if self.last_cmd is not None else (0.0, 0.0)

        # ✅ row도 idx,t_sec 제거 (헤더와 정합)
        self.csv_wr.writerow([
            rel.get('front_cam', ''), rel.get('rear_cam', ''),
            rel.get('left_cam', ''),  rel.get('right_cam', ''),
            f'{v:.6f}', f'{w:.6f}'
        ])

        self.frame_idx += 1
        self.img_buf.clear()

    def on_event(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(f'[COLLECTOR] bad /valet/event: {msg.data}')
            return

        typ = (payload.get('type') or '').upper().strip()

        if typ == 'EP_START':
            run_name = payload.get('run_name')
            episode_name = payload.get('episode_name')
            if not run_name or not episode_name:
                self.get_logger().warn('[COLLECTOR] EP_START missing run/episode')
                return
            if self.active:
                self._close_episode(delete_if_empty=False)
            self._open_episode(run_name, episode_name)

        elif typ == 'EP_END':
            self._close_episode(delete_if_empty=False)

    def on_record_control(self, msg: Bool):
        self.record_armed = bool(msg.data)

    def on_cmd(self, msg: Twist):
        self.last_cmd = (float(msg.linear.x), float(msg.angular.z))

    def on_img(self, cam: str, msg: CompressedImage):
        if not self.active:
            return
        t = stamp_to_sec(msg.header.stamp)
        self.img_buf[cam] = (t, bytes(msg.data))
        self._try_flush_frame()


def main(args=None):
    rclpy.init(args=args)
    node = CollectorNode()
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
