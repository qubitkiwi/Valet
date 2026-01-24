# valet_parking/launch/collect_stop.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='valet_parking',
            executable='stop_controller',
            output='screen',
            parameters=[{
                'dataset_root': '/home/ubuntu/ros2_ws/src/dataset/valet_parking',
                'debounce_sec': 0.25,
            }],
        ),

        Node(
            package='valet_parking',
            executable='collector_stop_node',
            output='screen',
            parameters=[{
                'dataset_root': '/home/ubuntu/ros2_ws/src/dataset/valet_parking',
                'cams': ['left_cam', 'right_cam', 'rear_cam', 'front_cam'],
                'sync_slop_sec': 0.10,
                'delete_empty_episode': False,
            }],
        ),
    ])
