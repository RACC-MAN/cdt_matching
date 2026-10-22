#ifndef CDT_MATCHING_ROS2__LOCALIZATION_NODE_HPP_
#define CDT_MATCHING_ROS2__LOCALIZATION_NODE_HPP_

#include <Eigen/Geometry>

#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "cdt_matching_ros2/image_matcher.hpp"
#include "cdt_matching_ros2/pose_optimizer.hpp"

namespace cdt_matching_ros2
{

class LocalizationNode : public rclcpp::Node
{
public:
  explicit LocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // --- 初期化 ---
  void declareParameters();
  bool loadCamera();
  bool loadMap();

  // --- コールバック ---
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);

  // --- 内部処理 ---
  Eigen::Isometry3d predictPose();
  void updateCameraExtrinsic();
  void publishPose(const rclcpp::Time & stamp, const Eigen::Matrix<double, 6, 6> & cov);
  void publishTf(const rclcpp::Time & stamp);
  void publishDebugImage(const rclcpp::Time & stamp);

  // --- パラメータ ---
  std::string global_frame_id_, odom_frame_id_, base_frame_id_, camera_frame_id_;
  std::string camera_yaml_path_, map_dir_;
  double map_voxel_leaf_ = 0.05;
  double image_scale_ = 0.25;
  double min_process_interval_ = 0.0;
  bool use_imu_ = false;
  bool use_odom_ = false;
  bool use_tf_camera_extrinsic_ = false;
  bool publish_debug_image_ = true;
  double score_threshold_ = 1.0;
  double tf_publish_rate_ = 20.0;
  double transform_tolerance_ = 0.1;
  Eigen::Isometry3d T_base_cam_ = Eigen::Isometry3d::Identity();
  bool camera_extrinsic_from_tf_done_ = false;

  // --- 状態 ---
  Eigen::Isometry3d pose_ = Eigen::Isometry3d::Identity();  // T_map_base
  bool initialized_ = false;
  bool map_ready_ = false;
  rclcpp::Time last_process_time_{0, 0, RCL_ROS_TIME};

  // odom
  bool have_odom_ = false;
  bool odom_prev_valid_ = false;
  Eigen::Isometry3d odom_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d odom_prev_ = Eigen::Isometry3d::Identity();

  // imu
  bool have_last_imu_ = false;
  rclcpp::Time last_imu_time_{0, 0, RCL_ROS_TIME};
  Eigen::Quaterniond imu_delta_ = Eigen::Quaterniond::Identity();

  // --- コンポーネント ---
  CameraModel camera_;
  bool camera_ready_ = false;
  ImageMatcher::Cloud::Ptr map_cloud_;
  std::shared_ptr<ImageMatcher> matcher_;
  std::shared_ptr<PoseOptimizer> optimizer_;
  ImageMatcherParams matcher_params_;
  OptimizerParams optimizer_params_;

  // --- ROS I/O ---
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;
  rclcpp::TimerBase::SharedPtr tf_timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace cdt_matching_ros2

#endif  // cdt_matching_ros2__LOCALIZATION_NODE_HPP_
