#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist


def now_str():
    return time.strftime("%H:%M:%S")


class MuxControllerDagger(Node):
    def __init__(self):
        super().__init__("mux_controller_dagger")

        # -------------------------
        # params (✅ 여기 토픽명만 "정정")
        # -------------------------
        self.declare_parameter("dagger", True)
        self.declare_parameter("resume_delay_sec", 2.5)

        # ✅ cmd_mux가 실제로 구독 중인 토픽: /mux/select (네 node info 기준)
        self.declare_parameter("select_topic", "/mux/select")

        # ✅ 너가 echo로 보고 있는 이벤트 토픽: /collector/event
        self.declare_parameter("event_out_topic", "/parking/event")

        self.declare_parameter("slot_out_topic", "/parking/slot_name")
        self.declare_parameter("stop_cmd_topic", "/controller/cmd_vel")

        self.dagger = bool(self.get_parameter("dagger").value)
        self.resume_delay = float(self.get_parameter("resume_delay_sec").value)

        self.select_topic = str(self.get_parameter("select_topic").value)
        self.event_out_topic = str(self.get_parameter("event_out_topic").value)
        self.slot_out_topic = str(self.get_parameter("slot_out_topic").value)
        self.stop_cmd_topic = str(self.get_parameter("stop_cmd_topic").value)

        # -------------------------
        # pubs/subs
        # -------------------------
        self.pub_select = self.create_publisher(String, self.select_topic, 10)
        self.pub_evt = self.create_publisher(String, self.event_out_topic, 10)
        self.pub_slot = self.create_publisher(String, self.slot_out_topic, 10)
        self.pub_stop = self.create_publisher(Twist, self.stop_cmd_topic, 10)

        self.sub_evt_raw = self.create_subscription(String, "/parking/event_raw", self._on_event_raw, 10)
        self.sub_joy_active = self.create_subscription(Bool, "/joy/active", self._on_joy_active, 10)

        self.timer = self.create_timer(0.05, self._tick)

        # -------------------------
        # state
        # -------------------------
        self.ep_active = False
        self.slot = 0
        self.slot_name = "p0"
        self.slot_selected = False

        self.mode = "stop"
        self.intervening = False

        self.pending_resume_at: Optional[float] = None
        self.last_joy_active = False

        self.sent_rec_on = False
        self.sent_rec_off = False

        # boot
        self._select_mode("stop")
        self._publish_slot()
        self._log_state(f"[BOOT] select_topic={self.select_topic} event_out={self.event_out_topic}")

    # -------------------------
    # helpers
    # -------------------------
    def _log(self, msg: str):
        self.get_logger().info(f"[{now_str()}] {msg}")

    def _log_state(self, tag: str):
        self._log(
            f"{tag} dagger={self.dagger} ep_active={self.ep_active} "
            f"slot={self.slot}({self.slot_name}) slot_selected={self.slot_selected} "
            f"intervening={self.intervening} mode={self.mode} "
            f"pending_resume_at={self.pending_resume_at}"
        )

    def _emit_event(self, payload: dict):
        payload = dict(payload)
        payload["dagger"] = bool(self.dagger)
        payload["ts"] = time.time()
        m = String()
        m.data = json.dumps(payload, ensure_ascii=False)
        self.pub_evt.publish(m)

    def _select_mode(self, mode: str):
        if mode not in ("stop", "joy", "policy"):
            return
        if mode == self.mode:
            return
        self.mode = mode

        m = String()
        m.data = mode
        self.pub_select.publish(m)

        if mode == "stop":
            z = Twist()
            z.linear.x = 0.0
            z.angular.z = 0.0
            self.pub_stop.publish(z)

        self._log(f"[MUX] -> {mode} (published to {self.select_topic})")

    def _publish_slot(self):
        m = String()
        m.data = self.slot_name
        self.pub_slot.publish(m)

    # -------------------------
    # event handlers
    # -------------------------
    def _on_event_raw(self, msg: String):
        try:
            ev = json.loads(msg.data)
        except Exception:
            self._log(f"[WARN] invalid event_raw: {msg.data}")
            return

        t = (ev.get("type") or "").upper().strip()

        if t == "EP_START":
            self._handle_ep_start()
        elif t == "EP_END":
            self._handle_ep_end()
        elif t == "SLOT":
            s = int(ev.get("slot", 0))
            self._handle_slot(s)
        else:
            self._log(f"[WARN] unknown event type: {t}")

    def _handle_ep_start(self):
        self.ep_active = True
        self.slot = 0
        self.slot_name = "p0"
        self.slot_selected = False

        self.intervening = False
        self.pending_resume_at = None
        self.sent_rec_on = False
        self.sent_rec_off = False

        self._select_mode("stop")
        self._publish_slot()

        self._log("[BTN] EP_START")
        self._emit_event({"type": "EP_START"})
        self._log_state("[EP_START] armed, waiting SLOT -> stop")
        if self.dagger:
            self._select_mode("stop")
        else:
            self._select_mode("joy")   # ✅ non-dagger는 사람 운전

    def _handle_slot(self, slot: int):
        if slot not in (1, 2, 3):
            self._log(f"[WARN] SLOT invalid: {slot}")
            return

        self.slot = slot
        self.slot_name = f"p{slot}"
        self.slot_selected = True
        self._publish_slot()

        self._log(f"[BTN] P{slot}")
        self._emit_event({"type": "SLOT", "slot": slot, "slot_name": self.slot_name})
        self._log_state(f"[SLOT] selected {slot} ({self.slot_name})")

        if (not self.dagger):
            # ✅ non-dagger는 항상 joy 유지 (policy 추론 금지)
            self._select_mode("joy")
            self._log_state("[SLOT] non-dagger -> joy")
            return

        # dagger=True일 때만 policy 자동 주행
        if self.ep_active and (not self.intervening) and (self.pending_resume_at is None):
            if self.dagger:
                self._select_mode("policy")
            else:
                self._select_mode("joy")


    def _handle_ep_end(self):
        self._log("[BTN] EP_END")

        if self.sent_rec_on and (not self.sent_rec_off):
            self._emit_event({"type": "REC_OFF"})
            self.sent_rec_off = True
            self._log("[REC] OFF forced by EP_END")

        self._emit_event({"type": "EP_END"})

        self.ep_active = False
        self.slot = 0
        self.slot_name = "p0"
        self.slot_selected = False

        self.intervening = False
        self.pending_resume_at = None
        self.sent_rec_on = False
        self.sent_rec_off = False

        self._select_mode("stop")
        self._publish_slot()
        self._log_state("[EP_END] -> stop & reset")

    def _on_joy_active(self, msg: Bool):
        active = bool(msg.data)

        if not self.ep_active:
            self.last_joy_active = active
            return

        if not self.slot_selected:
            self.last_joy_active = active
            return

        if active and (not self.last_joy_active):
            self.intervening = True
            self.pending_resume_at = None
            self._select_mode("joy")

            self._emit_event({"type": "REC_ON", "slot": self.slot, "slot_name": self.slot_name})
            self.sent_rec_on = True
            self.sent_rec_off = False
            self._log_state("[REC_ON] -> joy")

        if (not active) and self.last_joy_active:
            self.intervening = False

            if self.sent_rec_on and (not self.sent_rec_off):
                self._emit_event({"type": "REC_OFF"})
                self.sent_rec_off = True

            self._select_mode("stop")
            self.pending_resume_at = time.time() + self.resume_delay
            self._log_state("[REC_OFF_SCHEDULED] -> stop then resume")

        self.last_joy_active = active

    def _tick(self):
        if not self.ep_active or self.pending_resume_at is None:
            return
        if time.time() < self.pending_resume_at:
            return

        self.pending_resume_at = None

        if self.slot_selected and (not self.intervening):
            self._select_mode("policy" if self.dagger else "joy")
            self._log_state("[RESUME_POLICY] -> policy")
        else:
            self._select_mode("stop")
            self._log_state("[RESUME_POLICY] blocked -> stop")


def main(args=None):
    rclpy.init(args=args)
    node = MuxControllerDagger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
