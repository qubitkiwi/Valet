from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'valet_parking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sechankim',
    maintainer_email='sechankim98@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dagger_controller = valet_parking.dagger_controller:main',
            'collector_node = valet_parking.collector_node:main',        
            'stop_controller = valet_parking.stop_controller:main',
            'collector_stop_node = valet_parking.collector_stop_node:main',   
            'policy_infer_node = valet_parking.policy_infer_node:main',
        ],
    },
)
