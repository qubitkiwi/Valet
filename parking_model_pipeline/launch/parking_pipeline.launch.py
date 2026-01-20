from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    dagger = LaunchConfiguration('dagger')
    laptop_ip = LaunchConfiguration('laptop_ip')

    return LaunchDescription([
        DeclareLaunchArgument('dagger', default_value='true'),
        DeclareLaunchArgument('laptop_ip', default_value='192.168.0.10'),

        Node(
            package='parking_model_pipeline',
            executable='webcam_compressed_pub_parking',
            name='dual_cam_pub',
            output='screen',
            parameters=[
                {'front_dev': '/dev/front_cam'},
                {'rear_dev':  '/dev/rear_cam'},
                {'fps': 10.0},
                {'jpeg_quality': 50},
            ]
        ),

        # 조이스틱: 여기서는 invert 하지 말고 그대로(조향 부호는 mux에서만)
        Node(
            package='parking_model_pipeline',
            executable='joystick_event_publisher',
            name='joystick_event_publisher',
            output='screen',
            parameters=[
                {'max_linear': 0.25},
                {'max_angular': 3.0},          # <- steer_input_scale과 맞춰야 함
                {'js_dev': '/dev/input/js0'},
                {'deadzone': 0.1},
                {'active_hold_sec': 0.8},
                {'invert_linear': False},
                {'invert_angular': False},     # ✅ mux에서만 invert
            ]
        ),

        # 정책: 여기서도 invert 하지 말고 그대로(조향 부호는 mux에서만)
        Node(
            package='parking_model_pipeline',
            executable='policy_http_client_node',
            name='policy_http_client',
            output='screen',
            parameters=[
                {'server_url': ['http://', laptop_ip, ':8000']},
                {'rate_hz': 10.0},
                {'invert_linear': False},
                {'invert_angular': False},     # ✅ mux에서만 invert
            ]
        ),

        # ✅ Direct cmd_mux_node 버전 파라미터로 교체
        Node(
            package='parking_model_pipeline',
            executable='cmd_mux_node',
            name='cmd_mux',
            output='screen',
            parameters=[{
                'default': 'joy',
                'publish_hz': 20.0,

                # 속도 제한 (joy/policy 차등)
                'joy_out_max_linear': 0.25,
                'out_max_linear': 0.10,

                # v slew (Direct mux는 v만 필터링)
                'joy_slew_v': 10.0,
                'slew_v': 0.15,

                # 서보/차량
                'enable_servo': True,
                'wheelbase': 0.145,
                'servo_id': 3,
                'servo_center': 1500,
                'servo_scale': 2000.0,
                'max_steer_deg': 45.0,
                'servo_duration': 0.05,

                # ✅ Direct 조향 입력 해석
                'steer_input_normalized': False,  # /cmd_vel_* angular.z 가 [-3,+3]이면 False
                'steer_input_scale': 3.0,          # joystick max_angular와 동일
                'steer_deadzone': 0.05,

                # ✅ 조향 방향 반대면 여기만 바꾸면 됨
                'invert_steer': True,

                'debug_log': True,
            }]
        ),

        Node(
            package="parking_model_pipeline",
            executable="mux_controller_dagger",
            name="mux_controller_dagger",
            output="screen",
            parameters=[{
                "dagger": LaunchConfiguration("dagger"),
                "resume_delay_sec": 2.5,
                "select_topic": "/mux/select",
                "slot_out_topic": "/parking/slot_name",
                "stop_cmd_topic": "/controller/cmd_vel",
                "event_out_topic": "/parking/event",
            }],
        ),

        Node(
            package='parking_model_pipeline',
            executable='collector_http_bridge',
            name='collector_http_bridge',
            output='screen',
            parameters=[{
                'dagger': dagger,
                'collector_event_url': ['http://', laptop_ip, ':8000/collector/event'],
                'collector_frame_url': ['http://', laptop_ip, ':8000/collector/frame'],
                'send_hz': 10.0,
                'record_full_episode_when_not_dagger': True,
            }]
        ),
    ])
