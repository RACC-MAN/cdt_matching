import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ★ パッケージ名 (colcon でビルドしたパッケージ名に合わせる)
PKG = 'cdt_matching_ros2'


def generate_launch_description():
    pkg_share = get_package_share_directory(PKG)
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')
    default_rviz = os.path.join(pkg_share, 'config', 'localization.rviz')

    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = LaunchConfiguration('rviz_config')

    localization = Node(
        package=PKG,
        executable='localization_node',
        name='cdt_matching_ros2',
        output='screen',
        parameters=[params_file],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='log',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        localization,
        rviz,
    ])
