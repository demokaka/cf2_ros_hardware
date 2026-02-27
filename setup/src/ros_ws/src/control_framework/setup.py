from glob import glob

from setuptools import find_packages, setup

package_name = 'control_framework'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['control_framework/config/config.yaml']),
        ('share/' + package_name, ['launch/sitl.launch.py', 'launch/hitl.launch.py']),
        ('share/' + package_name + '/trajectory/trajectories', glob('control_framework/trajectory/trajectories/*.traj')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='sorinandres02@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "swarm = control_framework.entrypoints.swarm:main",
            "controller = control_framework.entrypoints.controller:main",
            "plotter = control_framework.entrypoints.plotter:main",
            "trajectory = control_framework.entrypoints.trajectory:main",
        ],
    },
)
