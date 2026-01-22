import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
# from driving_inference.msg import Recode
from std_msgs.msg import Int32

class DAggerNode(Node):
    def __init__(self):
        super().__init__('dagger_node')

        self.declare_parameter('joystick_cmd_topic', '/joystick/cmd_vel')
        self.declare_parameter('inference_cmd_topic', '/inference/cmd_vel')
        self.declare_parameter('recode_topic', '/record_control')
        self.declare_parameter('output_cmd_topic', '/controller/cmd_vel')

        self.joystick_cmd_topic = self.get_parameter('joystick_cmd_topic').get_parameter_value().string_value
        self.inference_cmd_topic = self.get_parameter('inference_cmd_topic').get_parameter_value().string_value
        self.recode_topic = self.get_parameter('recode_topic').get_parameter_value().string_value
        self.output_cmd_topic = self.get_parameter('output_cmd_topic').get_parameter_value().string_value

        # 구독자: 조이스틱 명령
        self.joystick_cmd_sub = self.create_subscription(
            Twist,
            self.joystick_cmd_topic,
            self.joystick_cmd_callback,
            10
        )

        # 구독자: 학습자 명령
        self.inference_cmd_sub = self.create_subscription(
            Twist,
            self.inference_cmd_topic,
            self.inference_cmd_callback,
            10
        )

        # 구독자: recode 토픽
        self.recode_sub = self.create_subscription(
            Int32,
            self.recode_topic,
            self.recode_callback,
            10
        )

        # 발행자: 최종 제어 명령
        self.cmd_pub = self.create_publisher(
            Twist,
            self.output_cmd_topic,
            10
        )

        self.use_expert = 0

    def joystick_cmd_callback(self, msg):
        if self.use_expert != 0:
            self.cmd_pub.publish(msg)

    def inference_cmd_callback(self, msg):
        if self.use_expert == 0:
            self.cmd_pub.publish(msg)

    def recode_callback(self, msg):
        self.use_expert = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = DAggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()