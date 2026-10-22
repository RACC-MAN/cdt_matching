#ifndef CDT_MATCHING_ROS2__POSE_OPTIMIZER_HPP_
#define CDT_MATCHING_ROS2__POSE_OPTIMIZER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <memory>
#include <vector>

#include "cdt_matching_ros2/image_matcher.hpp"

namespace cdt_matching_ros2
{

// NDT(pcl::NormalDistributionsTransform)の主要パラメータを踏襲:
//   max_iterations / transformation_epsilon / step_size
struct OptimizerParams
{
  int max_iterations = 20;
  double transformation_epsilon = 1e-3;  // 更新量ノルムがこれ未満で収束
  double step_size_trans = 0.2;          // 1反復の最大並進量 [m]    (NDTのstep_size相当)
  double step_size_rot = 0.1;            // 1反復の最大回転量 [rad]
  double lambda_init = 1e-2;             // LM減衰係数の初期値
  double lambda_factor = 4.0;
  double jacobian_eps_trans = 0.02;      // 数値微分の刻み [m]
  double jacobian_eps_rot = 0.01;        // 数値微分の刻み [rad]
  // 最適化する自由度 [x, y, z, rot_x, rot_y, rot_z] (map座標系軸まわり)
  std::array<bool, 6> dof_mask{{true, true, false, false, false, true}};
  double covariance_floor = 1e-4;        // 共分散の下限
};

struct OptimizeResult
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();  // T_map_base
  double score = 0.0;          // 0.5 * Σ r² (小さいほど良い)
  int iterations = 0;
  bool converged = false;
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();
};

class PoseOptimizer
{
public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  PoseOptimizer(std::shared_ptr<ImageMatcher> matcher, const OptimizerParams & params);

  // initial: 推定開始姿勢 T_map_base, T_base_cam: base_link→カメラ光学座標系
  // ※ 呼び出し前に matcher->setTargetImage / prepareLocalMap を済ませておくこと
  OptimizeResult optimize(
    const Eigen::Isometry3d & initial, const Eigen::Isometry3d & T_base_cam) const;

  static Eigen::Isometry3d applyDelta(const Eigen::Isometry3d & T, const Vector6d & delta);

private:
  double evaluate(
    const Eigen::Isometry3d & T_map_base, const Eigen::Isometry3d & T_base_cam,
    std::vector<double> & residuals) const;

  Eigen::MatrixXd computeJacobian(
    const Eigen::Isometry3d & T, const Eigen::Isometry3d & T_base_cam,
    const std::vector<double> & r0, const std::vector<int> & free_idx) const;

  std::shared_ptr<ImageMatcher> matcher_;
  OptimizerParams params_;
};

}  // namespace cdt_matching_ros2

#endif  // cdt_matching_ros2__POSE_OPTIMIZER_HPP_
