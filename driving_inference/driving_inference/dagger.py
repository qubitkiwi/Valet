import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Bool

class DAggerNode(Node):
    def __init__(self):
        super().__init__('dagger_node')

        self.declare_parameter('joystick_cmd_topic', '/joystick/cmd_vel')
        self.declare_parameter('inference_cmd_topic', '/driving/raw_cmd')
        self.declare_parameter('intervention_topic', '/human_intervention_state')
        self.declare_parameter('output_cmd_topic', '/controller/cmd_vel')

        self.joystick_topic = self.get_parameter('joystick_cmd_topic').value
        self.inference_topic = self.get_parameter('inference_cmd_topic').value
        self.intervention_topic = self.get_parameter('intervention_topic').value
        self.output_topic = self.get_parameter('output_cmd_topic').value

        # 조이스틱 구독
        self.joystick_sub = self.create_subscription(
            Twist, self.joystick_topic, self.joystick_callback, 10)

        # AI 모델 구독
        self.inference_sub = self.create_subscription(
            Twist, self.inference_topic, self.inference_callback, 10)

        # 개입 상태 구독
        self.human_sub = self.create_subscription(
            Bool, self.intervention_topic, self.intervention_callback, 10)

        # 최종 명령 발행
        self.cmd_pub = self.create_publisher(Twist, self.output_topic, 10)

        # 상태 변수: True이면 사람(Expert)이 제어, False이면 AI(Inference)가 제어
        self.is_human_control = True 

    def intervention_callback(self, msg):
        self.is_human_control = msg.data
        mode = "HUMAN (Joystick)" if self.is_human_control else "AI (Inference)"
        self.get_logger().info(f"Control Mode Switched to: {mode}")
        msg = Twist()
        self.cmd_pub.publish(msg)


    def joystick_callback(self, msg):
        if self.is_human_control:
            self.cmd_pub.publish(msg)

    def inference_callback(self, msg):
        if not self.is_human_control:
            self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DAggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()