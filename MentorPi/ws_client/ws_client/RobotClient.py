import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import asyncio
import websockets
import threading
import json
from functools import partial

SERVER_URL = "wss://ptwbmkhzpgkftzhe.tunnel.elice.io/ws/robot"

# 기존 카메라 토픽 (Index 0, 1, 2, 3)
TOPIC_LIST = [
    '/front_cam/image/compressed',
    '/rear_cam/image/compressed',
    '/left_cam/image/compressed',
    '/right_cam/image/compressed'
]

# 추가된 Safety 결과 토픽 (Index 4로 사용 예정)
SAFETY_TOPIC = '/safety/result/compressed'
SAFETY_INDEX = 4

class RobotClient(Node):
    def __init__(self):
        super().__init__('robot_ws_client')
        
        self.latest_frames = {}
        self.frame_flags = {}
        
        # 1. 기존 4방향 카메라 구독 (Index 0~3)
        for idx, topic in enumerate(TOPIC_LIST):
            self.latest_frames[idx] = None
            self.frame_flags[idx] = False
            self.create_subscription(
                CompressedImage, topic, partial(self.listener_callback, cam_index=idx), 10
            )

        # 2. Safety 결과 이미지 구독 (Index 4)
        self.latest_frames[SAFETY_INDEX] = None
        self.frame_flags[SAFETY_INDEX] = False
        self.create_subscription(
            CompressedImage, 
            SAFETY_TOPIC, 
            partial(self.listener_callback, cam_index=SAFETY_INDEX), 
            10
        )

        self.mode_publisher = self.create_publisher(String, '/robot_mode', 10)
        
        # 상태 정보 저장 (JSON String)
        self.current_status_json = ""
        self.status_updated = False
        self.create_subscription(String, '/robot_status', self.status_callback, 10)

        self.get_logger().info('Ready: Subscribing cameras, safety result & status...')

    def listener_callback(self, msg, cam_index):
        # 헤더(인덱스) + 이미지 바이너리 결합
        header = bytes([cam_index])
        self.latest_frames[cam_index] = header + bytes(msg.data)
        self.frame_flags[cam_index] = True
        
    def status_callback(self, msg):
        if self.current_status_json != msg.data:
            self.current_status_json = msg.data
            self.status_updated = True
        
    def publish_command(self, json_str):
        msg = String()
        msg.data = json_str
        self.mode_publisher.publish(msg)
        self.get_logger().info(f'Published Mode: {json_str}')

def ros_spin_thread(node):
    rclpy.spin(node)

async def run_client(node):
    print(f"🔗 서버 연결 시도: {SERVER_URL}")
    
    async with websockets.connect(SERVER_URL, ping_interval=None) as websocket:
        print("✅ 서버 연결됨!")
        
        while True:
            # 1. 영상 전송 (모든 등록된 프레임 키에 대해 반복)
            # 기존에는 range(len(TOPIC_LIST))였으나, 추가된 4번 인덱스도 포함하기 위해 keys() 사용
            for i in list(node.latest_frames.keys()):
                if node.frame_flags.get(i) and node.latest_frames.get(i):
                    try:
                        await websocket.send(node.latest_frames[i])
                        node.frame_flags[i] = False
                    except Exception as e:
                        print(f"Frame Send Error ({i}): {e}")
            
            # 2. 상태 전송 (JSON)
            if node.status_updated:
                try:
                    raw_data = json.loads(node.current_status_json)
                    payload = json.dumps({
                        "type": "status",
                        "data": raw_data
                    })
                    await websocket.send(payload)
                    node.status_updated = False
                except Exception as e:
                    print(f"Status Send Error: {e}")

            # 3. 명령 수신
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.005)
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        print(f"📩 명령 수신: {data['mode']}")
                        node.publish_command(data['mode'])
                    except json.JSONDecodeError: pass
            except asyncio.TimeoutError: pass
            except websockets.exceptions.ConnectionClosed:
                print("❌ 서버 연결 끊김"); break
            except Exception as e:
                print(f"⚠️ 에러: {e}"); await asyncio.sleep(1)

def main():
    rclpy.init()
    node = RobotClient()
    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()
    try: asyncio.run(run_client(node))
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()