import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String  # [NEW] 명령 발행용
import asyncio
import websockets
import threading
import json
from functools import partial

# ======================================================
# [설정] 서버 주소
SERVER_URL = "wss://ptwbmkhzpgkftzhe.tunnel.elice.io/ws/robot"

# [설정] 카메라 토픽 리스트
TOPIC_LIST = [
    '/front_cam/image/compressed', # 0: Front
    '/rear_cam/image/compressed',  # 1: Rear
    '/left_cam/image/compressed',  # 2: Left
    '/right_cam/image/compressed'  # 3: Right
]
# ======================================================

class RobotClient(Node):
    def __init__(self):
        super().__init__('robot_ws_client')
        
        # 1. 영상 구독 설정
        self.latest_frames = {}
        self.frame_flags = {}
        for idx, topic in enumerate(TOPIC_LIST):
            self.latest_frames[idx] = None
            self.frame_flags[idx] = False
            self.create_subscription(
                CompressedImage, topic, partial(self.listener_callback, cam_index=idx), 10
            )

        # 2. [NEW] 명령 발행 설정 (JSON 데이터를 String으로 보냄)
        self.mode_publisher = self.create_publisher(String, '/robot_mode', 10)
        self.get_logger().info('Ready: Publishing to /robot_mode, Subscribing images...')

    def listener_callback(self, msg, cam_index):
        # ID(1byte) + ImageBytes
        header = bytes([cam_index]) 
        self.latest_frames[cam_index] = header + bytes(msg.data)
        self.frame_flags[cam_index] = True
        
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
        print("✅ 서버 연결됨! (영상 전송 + 명령 수신 대기)")
        
        while True:
            # --- 1. 영상 전송 로직 (기존 동일) ---
            for i in range(len(TOPIC_LIST)):
                if node.frame_flags.get(i) and node.latest_frames.get(i):
                    try:
                        await websocket.send(node.latest_frames[i])
                        node.frame_flags[i] = False
                    except Exception:
                        pass
            
            # --- 2. [NEW] 명령 수신 로직 (JSON) ---
            try:
                # 0.005초 동안 메시지가 오는지 확인 (Non-blocking 효과)
                message = await asyncio.wait_for(websocket.recv(), timeout=0.005)
                
                # 메시지가 텍스트(JSON)라면 처리
                if isinstance(message, str):
                    try:
                        data = json.loads(message) # JSON 파싱 확인
                        print(f"📩 명령 수신: {data['mode']}")
                        
                        # ROS2 토픽으로 발행
#                        node.publish_command(message)
                        node.publish_command(data['mode'])
                    except json.JSONDecodeError:
                        print("JSON 형식이 아닙니다.")
                        
            except asyncio.TimeoutError:
                pass # 메시지 없으면 패스 (영상 계속 전송)
            except websockets.exceptions.ConnectionClosed:
                print("❌ 서버 연결 끊김")
                break
            except Exception as e:
                print(f"⚠️ 에러: {e}")
                await asyncio.sleep(1)

def main():
    rclpy.init()
    node = RobotClient()
    
    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        asyncio.run(run_client(node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
