from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='valet_parking',
            executable='dagger_controller',
            output='screen',
            parameters=[{
                'dagger': True,
                # JOY->POLICY 전환 조건(정지 체크)용
                'dataset_root': '/home/sechankim/ros2_ws/src/dataset/valet_parking',
                'back_to_policy_v_th': 0.02,
                'back_to_policy_w_th': 0.05,
            }],
        ),

        Node(
            package='valet_parking',
            executable='collector_node',
            output='screen',
            parameters=[{
                'dataset_root': '/home/sechankim/ros2_ws/src/dataset/valet_parking',
                'cams': ['left_cam', 'right_cam', 'rear_cam', 'front_cam'],
                'sync_slop_sec': 0.10,
                'moving_v_th': 0.02,
                'moving_w_th': 0.05,
            }],
        ),

        Node(
            package='valet_parking',
            executable='policy_infer_node',
            output='screen',
            parameters=[{
                'cams': ['front_cam', 'rear_cam', 'left_cam', 'right_cam'],
                'sync_slop_sec': 0.10,
                'publish_hz': 10.0,
                'backbone': 'v3s',
                # 학습된 best_model.pth 경로를 넣어라
                'ckpt_path': '/home/sechankim/ros2_ws/src/valet_parking/valet_parking/best_model.pth',
                'out_max_linear': 0.25,
                'out_max_angular': 2.0,
            }],
        ),
    ])
