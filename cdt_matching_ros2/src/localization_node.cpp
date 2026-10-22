#include "cdt_matching_ros2/localization_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace cdt_matching_ros2
{
namespace
{

Eigen::Isometry3d poseMsgToIso(const geometry_msgs::msg::Pose & p)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  if (q.norm() < 1e-9) {q = Eigen::Quaterniond::Identity();}
  q.normalize();
  T.linear() = q.toRotationMatrix();
  return T;
}

void isoToPoseMsg(const Eigen::Isometry3d & T, geometry_msgs::msg::Pose & p)
{
  p.position.x = T.translation().x();
  p.position.y = T.translation().y();
  p.position.z = T.translation().z();
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  p.orientation.w = q.w();
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
}

Eigen::Isometry3d transformMsgToIso(const geometry_msgs::msg::Transform & t)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(t.translation.x, t.translation.y, t.translation.z);
  Eigen::Quaterniond q(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z);
  q.normalize();
  T.linear() = q.toRotationMatrix();
  return T;
}

Eigen::Isometry3d xyzRpyToIso(const std::vector<double> & xyz, const std::vector<double> & rpy)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  if (xyz.size() >= 3) {T.translation() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);}
  if (rpy.size() >= 3) {
    T.linear() = (Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
  }
  return T;
}

}  // namespace

// ===========================================================================
LocalizationNode::LocalizationNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("cdt_matching_ros2", options)
{
  declareParameters();

  matcher_ = std::make_shared<ImageMatcher>(matcher_params_);
  optimizer_ = std::make_shared<PoseOptimizer>(matcher_, optimizer_params_);

  if (!loadCamera()) {
    throw std::runtime_error("failed to load camera yaml");
  }
  if (!loadMap()) {
    throw std::runtime_error("failed to load map pcd");
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  // --- Publishers ---
  pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/cur_pose", 10);
  map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "/map_cloud", rclcpp::QoS(1).transient_local().reliable());
  debug_img_pub_ = create_publisher<sensor_msgs::msg::Image>("/rendered_img", 1);

  // 地図を1回だけ(latched)配信
  sensor_msgs::msg::PointCloud2 map_msg;
  pcl::toROSMsg(*map_cloud_, map_msg);
  map_msg.header.frame_id = global_frame_id_;
  map_msg.header.stamp = now();
  map_pub_->publish(map_msg);

  // --- Subscribers ---
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "/input_img", rclcpp::SensorDataQoS(),
    std::bind(&LocalizationNode::imageCallback, this, std::placeholders::_1));
  initial_pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/initialpose", rclcpp::QoS(10),
    std::bind(&LocalizationNode::initialPoseCallback, this, std::placeholders::_1));
  if (use_imu_) {
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/input_imu", rclcpp::SensorDataQoS(),
      std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1));
  }
  if (use_odom_) {
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/input_odom", rclcpp::QoS(10),
      std::bind(&LocalizationNode::odomCallback, this, std::placeholders::_1));
  }

  // --- TFの定期配信 (推定が来ない間も TF を途切れさせない) ---
  if (tf_publish_rate_ > 0.0) {
    tf_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / tf_publish_rate_),
      [this]() {
        if (initialized_) {
          publishTf(now() + rclcpp::Duration::from_seconds(transform_tolerance_));
        }
      });
  }

  // --- 初期位置 (パラメータ指定) ---
  if (get_parameter("set_initial_pose").as_bool()) {
    const std::vector<double> xyz{
      get_parameter("initial_pose_x").as_double(),
      get_parameter("initial_pose_y").as_double(),
      get_parameter("initial_pose_z").as_double()};
    const std::vector<double> rpy{
      get_parameter("initial_pose_roll").as_double(),
      get_parameter("initial_pose_pitch").as_double(),
      get_parameter("initial_pose_yaw").as_double()};
    pose_ = xyzRpyToIso(xyz, rpy);
    initialized_ = true;
    RCLCPP_INFO(get_logger(), "initial pose set from parameters");
  } else {
    RCLCPP_INFO(get_logger(), "waiting for /initialpose (RViz2 '2D Pose Estimate')");
  }
}

// ===========================================================================
void LocalizationNode::declareParameters()
{
  global_frame_id_ = declare_parameter<std::string>("global_frame_id", "map");
  odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
  base_frame_id_ = declare_parameter<std::string>("base_frame_id", "base_link");
  camera_frame_id_ = declare_parameter<std::string>("camera_frame_id", "camera_optical_frame");
  camera_yaml_path_ = declare_parameter<std::string>("camera_yaml_path", "");
  map_dir_ = declare_parameter<std::string>("map_dir", "");
  map_voxel_leaf_ = declare_parameter<double>("map_voxel_leaf", 0.05);
  image_scale_ = declare_parameter<double>("image_scale", 0.25);
  min_process_interval_ = declare_parameter<double>("min_process_interval", 0.0);
  use_imu_ = declare_parameter<bool>("use_imu", false);
  use_odom_ = declare_parameter<bool>("use_odom", false);
  use_tf_camera_extrinsic_ = declare_parameter<bool>("use_tf_camera_extrinsic", false);
  publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
  score_threshold_ = declare_parameter<double>("score_threshold", 1.0);
  tf_publish_rate_ = declare_parameter<double>("tf_publish_rate", 20.0);
  transform_tolerance_ = declare_parameter<double>("transform_tolerance", 0.1);

  // base_link → カメラ光学座標系 (z前方, x右, y下)。既定値は「前向き・高さ0.1m」のカメラ
  const auto xyz = declare_parameter<std::vector<double>>(
    "base_to_camera_xyz", std::vector<double>{0.0, 0.0, 0.1});
  const auto rpy = declare_parameter<std::vector<double>>(
    "base_to_camera_rpy", std::vector<double>{-M_PI / 2.0, 0.0, -M_PI / 2.0});
  T_base_cam_ = xyzRpyToIso(xyz, rpy);

  // 初期位置
  declare_parameter<bool>("set_initial_pose", false);
  declare_parameter<double>("initial_pose_x", 0.0);
  declare_parameter<double>("initial_pose_y", 0.0);
  declare_parameter<double>("initial_pose_z", 0.0);
  declare_parameter<double>("initial_pose_roll", 0.0);
  declare_parameter<double>("initial_pose_pitch", 0.0);
  declare_parameter<double>("initial_pose_yaw", 0.0);

  // --- 画像マッチング ---
  auto & mp = matcher_params_;
  mp.color_levels = declare_parameter<int>("color_levels", 8);
  mp.min_class_pixels = declare_parameter<int>("min_class_pixels", 10);
  mp.weight_ratio = declare_parameter<double>("weight_ratio", 1.0);
  mp.weight_centroid = declare_parameter<double>("weight_centroid", 1.0);
  mp.weight_stddev = declare_parameter<double>("weight_stddev", 1.0);
  mp.weight_gradient = declare_parameter<double>("weight_gradient", 1.0);
  mp.weight_coverage = declare_parameter<double>("weight_coverage", 0.5);
  mp.min_coverage = declare_parameter<double>("min_coverage", 0.02);
  mp.empty_render_penalty = declare_parameter<double>("empty_render_penalty", 1.0);
  mp.min_range = declare_parameter<double>("render_min_range", 0.1);
  mp.max_range = declare_parameter<double>("render_max_range", 15.0);
  mp.point_size = declare_parameter<double>("render_point_size", 0.03);
  mp.max_splat_px = declare_parameter<int>("render_max_splat_px", 3);
  mp.local_map_margin = declare_parameter<double>("local_map_margin", 2.0);

  // --- 最適化 (NDT準拠) ---
  auto & op = optimizer_params_;
  op.max_iterations = declare_parameter<int>("max_iterations", 20);
  op.transformation_epsilon = declare_parameter<double>("transformation_epsilon", 1e-3);
  op.step_size_trans = declare_parameter<double>("step_size", 0.2);
  op.step_size_rot = declare_parameter<double>("step_size_rot", 0.1);
  op.lambda_init = declare_parameter<double>("lambda_init", 1e-2);
  op.jacobian_eps_trans = declare_parameter<double>("jacobian_eps_trans", 0.02);
  op.jacobian_eps_rot = declare_parameter<double>("jacobian_eps_rot", 0.01);
  op.covariance_floor = declare_parameter<double>("covariance_floor", 1e-4);
  const auto dof = declare_parameter<std::vector<bool>>(
    "optimize_dof_mask", std::vector<bool>{true, true, false, false, false, true});
  for (size_t i = 0; i < 6 && i < dof.size(); ++i) {
    op.dof_mask[i] = dof[i];
  }
}

bool LocalizationNode::loadCamera()
{
  std::string err;
  if (!camera_.loadFromYaml(camera_yaml_path_, err)) {
    RCLCPP_FATAL(get_logger(), "%s", err.c_str());
    return false;
  }
  RCLCPP_INFO(get_logger(), "camera yaml loaded: %s", camera_yaml_path_.c_str());
  return true;
}

bool LocalizationNode::loadMap()
{
  namespace fs = std::filesystem;
  std::vector<std::string> files;
  if (fs::is_regular_file(map_dir_)) {
    files.push_back(map_dir_);
  } else if (fs::is_directory(map_dir_)) {
    for (const auto & e : fs::directory_iterator(map_dir_)) {
      if (e.is_regular_file() && e.path().extension() == ".pcd") {
        files.push_back(e.path().string());
      }
    }
    std::sort(files.begin(), files.end());
  }
  if (files.empty()) {
    RCLCPP_FATAL(get_logger(), "no .pcd found in map_dir: %s", map_dir_.c_str());
    return false;
  }

  auto merged = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  for (const auto & f : files) {
    pcl::PointCloud<pcl::PointXYZRGB> c;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(f, c) < 0) {
      RCLCPP_ERROR(get_logger(), "failed to load %s", f.c_str());
      continue;
    }
    *merged += c;
    RCLCPP_INFO(get_logger(), "loaded %s (%zu points)", f.c_str(), c.size());
  }
  if (merged->empty()) {return false;}

  map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (map_voxel_leaf_ > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(merged);
    vg.setLeafSize(
      static_cast<float>(map_voxel_leaf_), static_cast<float>(map_voxel_leaf_),
      static_cast<float>(map_voxel_leaf_));
    vg.filter(*map_cloud_);
  } else {
    map_cloud_ = merged;
  }
  RCLCPP_INFO(get_logger(), "map points: %zu (after voxel filter)", map_cloud_->size());

  matcher_->setMap(map_cloud_);
  map_ready_ = true;
  return true;
}

// ===========================================================================
void LocalizationNode::updateCameraExtrinsic()
{
  if (!use_tf_camera_extrinsic_ || camera_extrinsic_from_tf_done_) {return;}
  try {
    const auto tf = tf_buffer_->lookupTransform(
      base_frame_id_, camera_frame_id_, tf2::TimePointZero);
    T_base_cam_ = transformMsgToIso(tf.transform);
    camera_extrinsic_from_tf_done_ = true;
    RCLCPP_INFO(get_logger(), "camera extrinsic loaded from TF (%s -> %s)",
      base_frame_id_.c_str(), camera_frame_id_.c_str());
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "TF camera extrinsic not available yet (%s). Using params.",
      e.what());
  }
}

void LocalizationNode::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  const rclcpp::Time t(msg->header.stamp);
  if (have_last_imu_) {
    const double dt = (t - last_imu_time_).seconds();
    if (dt > 0.0 && dt < 0.5) {
      // IMU座標系 = base_link と仮定 (異なる場合は上流でTF変換しておく)
      const Eigen::Vector3d w(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
      const double ang = w.norm() * dt;
      if (ang > 1e-9) {
        imu_delta_ = imu_delta_ * Eigen::Quaterniond(Eigen::AngleAxisd(ang, w.normalized()));
        imu_delta_.normalize();
      }
    }
  }
  last_imu_time_ = t;
  have_last_imu_ = true;
}

void LocalizationNode::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  odom_pose_ = poseMsgToIso(msg->pose.pose);
  have_odom_ = true;
}

void LocalizationNode::initialPoseCallback(
  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  Eigen::Isometry3d T = poseMsgToIso(msg->pose.pose);

  // 指定フレームが global_frame と異なる場合は TF で変換
  if (!msg->header.frame_id.empty() && msg->header.frame_id != global_frame_id_) {
    try {
      const auto tf = tf_buffer_->lookupTransform(
        global_frame_id_, msg->header.frame_id, tf2::TimePointZero);
      T = transformMsgToIso(tf.transform) * T;
    } catch (const tf2::TransformException & e) {
      RCLCPP_ERROR(get_logger(), "initialpose transform failed: %s", e.what());
      return;
    }
  }
  pose_ = T;
  initialized_ = true;
  // 予測用の積算状態をリセット
  odom_prev_valid_ = false;
  imu_delta_ = Eigen::Quaterniond::Identity();
  RCLCPP_INFO(get_logger(), "initial pose reset: x=%.2f y=%.2f z=%.2f",
    pose_.translation().x(), pose_.translation().y(), pose_.translation().z());
  publishTf(now() + rclcpp::Duration::from_seconds(transform_tolerance_));
}

// ===========================================================================
Eigen::Isometry3d LocalizationNode::predictPose()
{
  Eigen::Isometry3d pred = pose_;

  if (use_odom_ && have_odom_) {
    if (odom_prev_valid_) {
      Eigen::Isometry3d delta = odom_prev_.inverse() * odom_pose_;
      if (use_imu_) {
        // 並進はオドメトリ、回転はIMU(ジャイロ積分)を採用
        delta.linear() = imu_delta_.toRotationMatrix();
      }
      pred = pose_ * delta;
    }
    odom_prev_ = odom_pose_;
    odom_prev_valid_ = true;
  } else if (use_imu_) {
    pred.linear() = pose_.linear() * imu_delta_.toRotationMatrix();
  }
  imu_delta_ = Eigen::Quaterniond::Identity();
  return pred;
}

void LocalizationNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  if (!map_ready_ || !initialized_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for initial pose");
    return;
  }
  const rclcpp::Time stamp(msg->header.stamp);
  if (min_process_interval_ > 0.0 && last_process_time_.nanoseconds() > 0 &&
    (stamp - last_process_time_).seconds() < min_process_interval_)
  {
    return;
  }
  last_process_time_ = stamp;

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
    return;
  }

  if (!camera_ready_) {
    camera_.configure(cv_ptr->image.size(), image_scale_);
    matcher_->setCameraIntrinsics(
      camera_.fx(), camera_.fy(), camera_.cx(), camera_.cy(), camera_.procSize());
    camera_ready_ = true;
    RCLCPP_INFO(get_logger(), "processing resolution: %dx%d",
      camera_.procSize().width, camera_.procSize().height);
  }
  updateCameraExtrinsic();

  const cv::Mat img = camera_.preprocess(cv_ptr->image);
  const Eigen::Isometry3d pred = predictPose();

  matcher_->setTargetImage(img);
  matcher_->prepareLocalMap((pred * T_base_cam_).translation());
  const OptimizeResult res = optimizer_->optimize(pred, T_base_cam_);

  if (res.score <= score_threshold_) {
    pose_ = res.pose;
  } else {
    pose_ = pred;  // 信頼できないため予測(デッドレコニング)を採用
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "match score %.4f > threshold %.4f : optimization rejected", res.score, score_threshold_);
  }
  RCLCPP_DEBUG(get_logger(), "score=%.4f iter=%d converged=%d",
    res.score, res.iterations, res.converged);

  publishPose(stamp, res.covariance);
  publishTf(now() + rclcpp::Duration::from_seconds(transform_tolerance_));
  if (publish_debug_image_) {publishDebugImage(stamp);}
}

// ===========================================================================
void LocalizationNode::publishPose(
  const rclcpp::Time & stamp, const Eigen::Matrix<double, 6, 6> & cov)
{
  geometry_msgs::msg::PoseWithCovarianceStamped out;
  out.header.stamp = stamp;
  out.header.frame_id = global_frame_id_;
  isoToPoseMsg(pose_, out.pose.pose);
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out.pose.covariance[r * 6 + c] = cov(r, c);
    }
  }
  pose_pub_->publish(out);
}

void LocalizationNode::publishTf(const rclcpp::Time & stamp)
{
  // lidar_localization_ros2 と同様:
  //   odom 使用時 : map -> odom を配信 (odom -> base_link はオドメトリ側が配信)
  //   odom 不使用 : map -> base_link を直接配信
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp;
  tf.header.frame_id = global_frame_id_;

  Eigen::Isometry3d T;
  if (use_odom_ && have_odom_) {
    T = pose_ * odom_pose_.inverse();
    tf.child_frame_id = odom_frame_id_;
  } else {
    T = pose_;
    tf.child_frame_id = base_frame_id_;
  }
  tf.transform.translation.x = T.translation().x();
  tf.transform.translation.y = T.translation().y();
  tf.transform.translation.z = T.translation().z();
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  tf.transform.rotation.w = q.w();
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  tf_broadcaster_->sendTransform(tf);
}

void LocalizationNode::publishDebugImage(const rclcpp::Time & stamp)
{
  if (debug_img_pub_->get_subscription_count() == 0) {return;}
  RenderResult rr;
  matcher_->render(pose_ * T_base_cam_, rr);
  std_msgs::msg::Header header;
  header.stamp = stamp;
  header.frame_id = camera_frame_id_;
  debug_img_pub_->publish(*cv_bridge::CvImage(header, "bgr8", rr.bgr).toImageMsg());
}

}  // namespace cdt_matching_ros2
