from setuptools import find_packages, setup

package_name = 'px505_controller'

import os
from glob import glob

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ["px505_controller/config/config.yaml"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "controller = px505_controller.main:main",
            "plotter = px505_controller.plotter:main",
            "action_publisher = px505_controller.action_publisher:main",
        ],
    },
)
