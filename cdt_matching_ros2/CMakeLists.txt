cmake_minimum_required(VERSION 3.8)
project(cdt_matching_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

add_definitions(${PCL_DEFINITIONS})

# ---- 画像マッチング (ROS非依存) ----
add_library(image_matcher STATIC src/image_matcher.cpp)
target_include_directories(image_matcher PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_matcher PUBLIC Eigen3::Eigen ${OpenCV_LIBS} ${PCL_LIBRARIES})

# ---- 位置最適化 (image_matcher.hpp を include) ----
add_library(pose_optimizer STATIC src/pose_optimizer.cpp)
target_link_libraries(pose_optimizer PUBLIC image_matcher)

# ---- ROS2 ノード ----
add_executable(localization_node src/main.cpp src/localization_node.cpp)
target_link_libraries(localization_node pose_optimizer image_matcher)
ament_target_dependencies(localization_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros cv_bridge pcl_conversions)

install(TARGETS localization_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
