from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='valet_parking',
            executable='dagger_controller',
            output='screen',
            parameters=[{'dagger': False}],
        ),
        Node(
            package='valet_parking',
            executable='collector_node',
            output='screen',
        ),
    ])
