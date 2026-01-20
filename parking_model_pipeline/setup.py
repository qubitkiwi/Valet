from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'parking_model_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='11306260+liangfuyuan@user.noreply.gitee.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_compressed_pub_parking = parking_model_pipeline.webcam_compressed_pub_parking:main',
            'joystick_event_publisher = parking_model_pipeline.joystick_event_publisher:main',
            'policy_http_client_node = parking_model_pipeline.policy_http_client_node:main',
            'cmd_mux_node = parking_model_pipeline.cmd_mux_node:main',
            'mux_controller_dagger = parking_model_pipeline.mux_controller_dagger:main',
            'collector_http_bridge = parking_model_pipeline.collector_http_bridge:main',
        ],
    },
)
