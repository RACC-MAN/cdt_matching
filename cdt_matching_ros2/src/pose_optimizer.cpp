#include "cdt_matching_ros2/pose_optimizer.hpp"

#include <algorithm>
#include <cmath>

namespace cdt_matching_ros2
{

PoseOptimizer::PoseOptimizer(std::shared_ptr<ImageMatcher> matcher, const OptimizerParams & params)
: matcher_(std::move(matcher)), params_(params) {}

Eigen::Isometry3d PoseOptimizer::applyDelta(const Eigen::Isometry3d & T, const Vector6d & d)
{
  Eigen::Isometry3d out = T;
  out.translation() += d.head<3>();
  const Eigen::Vector3d w = d.tail<3>();
  const double ang = w.norm();
  if (ang > 1e-12) {
    const Eigen::Matrix3d dR = Eigen::AngleAxisd(ang, w / ang).toRotationMatrix();
    out.linear() = dR * T.linear();  // map座標系まわりの回転
  }
  return out;
}

double PoseOptimizer::evaluate(
  const Eigen::Isometry3d & T_map_base, const Eigen::Isometry3d & T_base_cam,
  std::vector<double> & residuals) const
{
  matcher_->computeResiduals(T_map_base * T_base_cam, residuals);
  double s = 0.0;
  for (double v : residuals) {s += v * v;}
  return 0.5 * s;
}

Eigen::MatrixXd PoseOptimizer::computeJacobian(
  const Eigen::Isometry3d & T, const Eigen::Isometry3d & T_base_cam,
  const std::vector<double> & r0, const std::vector<int> & free_idx) const
{
  const int m = static_cast<int>(r0.size());
  const int n = static_cast<int>(free_idx.size());
  Eigen::MatrixXd J(m, n);
  std::vector<double> rj;
  for (int k = 0; k < n; ++k) {
    const int dof = free_idx[k];
    const double eps = dof < 3 ? params_.jacobian_eps_trans : params_.jacobian_eps_rot;
    Vector6d d = Vector6d::Zero();
    d[dof] = eps;
    evaluate(applyDelta(T, d), T_base_cam, rj);
    for (int i = 0; i < m; ++i) {
      J(i, k) = (rj[i] - r0[i]) / eps;
    }
  }
  return J;
}

OptimizeResult PoseOptimizer::optimize(
  const Eigen::Isometry3d & initial, const Eigen::Isometry3d & T_base_cam) const
{
  OptimizeResult out;
  out.pose = initial;

  std::vector<int> free_idx;
  for (int i = 0; i < 6; ++i) {
    if (params_.dof_mask[i]) {free_idx.push_back(i);}
  }
  const int n = static_cast<int>(free_idx.size());

  Eigen::Isometry3d T = initial;
  std::vector<double> r;
  double cost = evaluate(T, T_base_cam, r);
  out.score = cost;
  if (n == 0) {
    out.converged = true;
    return out;
  }
  const int m = static_cast<int>(r.size());

  double lambda = params_.lambda_init;
  std::vector<double> r_new;

  // Levenberg-Marquardt (NDTのNewton法+ステップ制限に相当)
  for (int iter = 0; iter < params_.max_iterations; ++iter) {
    out.iterations = iter + 1;

    const Eigen::MatrixXd J = computeJacobian(T, T_base_cam, r, free_idx);
    const Eigen::Map<const Eigen::VectorXd> rv(r.data(), m);
    const Eigen::MatrixXd H = J.transpose() * J;
    const Eigen::VectorXd g = J.transpose() * rv;

    bool accepted = false;
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);
    Eigen::Isometry3d T_new = T;
    double cost_new = cost;

    for (int inner = 0; inner < 8; ++inner) {
      Eigen::MatrixXd A = H;
      for (int k = 0; k < n; ++k) {
        A(k, k) += lambda * std::max(H(k, k), 1e-9);
      }
      delta = A.ldlt().solve(-g);

      // ステップ幅制限
      double factor = 1.0;
      for (int k = 0; k < n; ++k) {
        const double lim = free_idx[k] < 3 ? params_.step_size_trans : params_.step_size_rot;
        factor = std::max(factor, std::abs(delta[k]) / lim);
      }
      delta /= factor;

      Vector6d d6 = Vector6d::Zero();
      for (int k = 0; k < n; ++k) {d6[free_idx[k]] = delta[k];}
      T_new = applyDelta(T, d6);
      cost_new = evaluate(T_new, T_base_cam, r_new);

      if (cost_new < cost) {
        accepted = true;
        lambda = std::max(lambda / params_.lambda_factor, 1e-9);
        break;
      }
      lambda *= params_.lambda_factor;
    }

    if (!accepted) {  // これ以上改善できない
      out.converged = true;
      break;
    }
    T = T_new;
    cost = cost_new;
    r = r_new;

    if (delta.norm() < params_.transformation_epsilon) {
      out.converged = true;
      break;
    }
  }

  out.pose = T;
  out.score = cost;

  // 共分散の近似: σ² (JᵀJ)^-1
  {
    const Eigen::MatrixXd J = computeJacobian(T, T_base_cam, r, free_idx);
    Eigen::MatrixXd H = J.transpose() * J;
    H += 1e-9 * Eigen::MatrixXd::Identity(n, n);
    const double sigma2 = std::max(2.0 * cost / std::max(m - n, 1), 1e-8);
    const Eigen::MatrixXd Cf = sigma2 * H.inverse();

    Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Zero();
    for (int i = 0; i < 6; ++i) {C(i, i) = 1e-6;}  // 固定自由度
    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < n; ++b) {
        C(free_idx[a], free_idx[b]) = Cf(a, b);
      }
      C(free_idx[a], free_idx[a]) = std::max(C(free_idx[a], free_idx[a]), params_.covariance_floor);
    }
    out.covariance = C;
  }
  return out;
}

}  // namespace cdt_matching_ros2
