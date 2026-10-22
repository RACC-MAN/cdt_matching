#include "cdt_matching_ros2/image_matcher.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cdt_matching_ros2
{

// ===========================================================================
// CameraModel
// ===========================================================================
bool CameraModel::loadFromYaml(const std::string & path, std::string & error)
{
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    error = "cannot open camera yaml: " + path;
    return false;
  }
  cv::Mat K, D;
  int w = 0, h = 0;
  fs["camera_matrix"] >> K;
  fs["distortion_coefficients"] >> D;
  fs["image_width"] >> w;
  fs["image_height"] >> h;

  if (K.empty() || K.rows != 3 || K.cols != 3) {
    error = "camera_matrix (3x3, opencv-matrix) not found in " + path;
    return false;
  }
  K.convertTo(K_calib_, CV_64F);
  if (D.empty()) {
    dist_ = cv::Mat::zeros(1, 5, CV_64F);
  } else {
    D.convertTo(dist_, CV_64F);
  }
  if (w > 0 && h > 0) {
    calib_size_ = cv::Size(w, h);
  } else {
    // サイズ記載なし → 主点から推定
    calib_size_ = cv::Size(
      static_cast<int>(std::lround(K_calib_.at<double>(0, 2) * 2.0)),
      static_cast<int>(std::lround(K_calib_.at<double>(1, 2) * 2.0)));
  }
  loaded_ = true;
  configured_ = false;
  return true;
}

void CameraModel::configure(const cv::Size & input_size, double scale)
{
  proc_size_ = cv::Size(
    std::max(1, static_cast<int>(std::lround(input_size.width * scale))),
    std::max(1, static_cast<int>(std::lround(input_size.height * scale))));
  if (calib_size_.width <= 0 || calib_size_.height <= 0) {
    calib_size_ = input_size;
  }
  const double sx = static_cast<double>(proc_size_.width) / calib_size_.width;
  const double sy = static_cast<double>(proc_size_.height) / calib_size_.height;

  cv::Mat K = K_calib_.clone();
  K.at<double>(0, 0) *= sx;
  K.at<double>(0, 2) *= sx;
  K.at<double>(1, 1) *= sy;
  K.at<double>(1, 2) *= sy;

  // 歪み係数は正規化座標系で定義されるため解像度が変わってもそのまま使える
  cv::initUndistortRectifyMap(K, dist_, cv::Mat(), K, proc_size_, CV_16SC2, map1_, map2_);

  fx_ = K.at<double>(0, 0);
  fy_ = K.at<double>(1, 1);
  cx_ = K.at<double>(0, 2);
  cy_ = K.at<double>(1, 2);
  configured_ = true;
}

cv::Mat CameraModel::preprocess(const cv::Mat & bgr) const
{
  cv::Mat small;
  if (bgr.size() != proc_size_) {
    cv::resize(bgr, small, proc_size_, 0, 0, cv::INTER_AREA);
  } else {
    small = bgr;
  }
  cv::Mat out;
  cv::remap(small, out, map1_, map2_, cv::INTER_LINEAR);
  return out;
}

// ===========================================================================
// ColorFeatureExtractor
// ===========================================================================
ColorFeatureExtractor::ColorFeatureExtractor(int levels, int min_pixels)
: levels_(std::clamp(levels, 1, 256)), min_pixels_(std::max(1, min_pixels)) {}

ImageFeatures ColorFeatureExtractor::extract(const cv::Mat & bgr, const cv::Mat & mask) const
{
  struct Acc
  {
    double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  };

  ImageFeatures feat;
  feat.levels = levels_;
  feat.classes.assign(3 * levels_, ClassFeature());
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    return feat;
  }

  std::vector<Acc> acc(3 * levels_);
  int lut[256];
  for (int v = 0; v < 256; ++v) {
    lut[v] = (v * levels_) >> 8;  // 0 .. levels-1
  }

  const bool use_mask = !mask.empty();
  int total = 0;
  for (int y = 0; y < bgr.rows; ++y) {
    const uchar * p = bgr.ptr<uchar>(y);
    const uchar * m = use_mask ? mask.ptr<uchar>(y) : nullptr;
    const double yd = y;
    for (int x = 0; x < bgr.cols; ++x) {
      if (m && m[x] == 0) {continue;}
      ++total;
      const double xd = x;
      for (int c = 0; c < 3; ++c) {
        Acc & a = acc[c * levels_ + lut[p[3 * x + c]]];
        a.n += 1.0;
        a.sx += xd;
        a.sy += yd;
        a.sxx += xd * xd;
        a.syy += yd * yd;
        a.sxy += xd * yd;
      }
    }
  }
  feat.total_pixels = total;
  if (total == 0) {return feat;}

  const double s = static_cast<double>(std::max(bgr.cols, bgr.rows));  // 等方正規化
  for (size_t k = 0; k < acc.size(); ++k) {
    const Acc & a = acc[k];
    if (a.n < min_pixels_) {continue;}
    ClassFeature & f = feat.classes[k];
    const double mx = a.sx / a.n;
    const double my = a.sy / a.n;
    const double vxx = std::max(0.0, a.sxx / a.n - mx * mx);
    const double vyy = std::max(0.0, a.syy / a.n - my * my);
    const double vxy = a.sxy / a.n - mx * my;

    f.valid = true;
    f.count = static_cast<int>(a.n);
    f.ratio = a.n / total;
    f.centroid = Eigen::Vector2d(mx / s, my / s);
    f.stddev = Eigen::Vector2d(std::sqrt(vxx) / s, std::sqrt(vyy) / s);

    // 共分散行列の主軸 = 総最小二乗による近似直線の方向
    const double d = vxx - vyy;
    const double r = std::sqrt(d * d + 4.0 * vxy * vxy);
    f.slope_angle = 0.5 * std::atan2(2.0 * vxy, d);
    f.elongation = r / (vxx + vyy + 1e-9);
    f.gradient = f.elongation *
      Eigen::Vector2d(std::cos(2.0 * f.slope_angle), std::sin(2.0 * f.slope_angle));
  }
  return feat;
}

// ===========================================================================
// FeatureComparator
// ===========================================================================
void FeatureComparator::compute(
  const ImageFeatures & rendered, const ImageFeatures & target,
  double coverage, std::vector<double> & res) const
{
  const size_t K = rendered.classes.size();
  res.assign(K * kResidualsPerClass + 1, 0.0);

  for (size_t k = 0; k < K; ++k) {
    const ClassFeature & a = rendered.classes[k];
    const ClassFeature & b = target.classes[k];
    if (!a.valid && !b.valid) {continue;}

    double * r = &res[k * kResidualsPerClass];
    const double ra = a.valid ? a.ratio : 0.0;
    const double rb = b.valid ? b.ratio : 0.0;
    r[0] = params_.weight_ratio * (ra - rb);

    // 両方に存在するクラスのみ幾何特徴を比較 (占有率が大きいクラスほど信頼)
    if (a.valid && b.valid) {
      const double wk = std::sqrt(0.5 * (ra + rb));
      const Eigen::Vector2d dc = a.centroid - b.centroid;
      const Eigen::Vector2d ds = a.stddev - b.stddev;
      const Eigen::Vector2d dg = a.gradient - b.gradient;
      r[1] = wk * params_.weight_centroid * dc.x();
      r[2] = wk * params_.weight_centroid * dc.y();
      r[3] = wk * params_.weight_stddev * ds.x();
      r[4] = wk * params_.weight_stddev * ds.y();
      r[5] = wk * params_.weight_gradient * dg.x();
      r[6] = wk * params_.weight_gradient * dg.y();
    }
  }

  double cov_res = params_.weight_coverage * (1.0 - coverage);
  if (coverage < params_.min_coverage) {
    cov_res += params_.empty_render_penalty;
  }
  res.back() = cov_res;
}

// ===========================================================================
// ImageMatcher
// ===========================================================================
ImageMatcher::ImageMatcher(const ImageMatcherParams & params)
: params_(params),
  extractor_(params.color_levels, params.min_class_pixels),
  comparator_(params) {}

void ImageMatcher::setCameraIntrinsics(
  double fx, double fy, double cx, double cy, const cv::Size & size)
{
  fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy; size_ = size;
}

void ImageMatcher::setMap(const Cloud::ConstPtr & map)
{
  map_ = map;
  local_pts_.clear();
  local_cols_.clear();
}

void ImageMatcher::setTargetImage(const cv::Mat & bgr)
{
  target_ = bgr;
}

void ImageMatcher::prepareLocalMap(const Eigen::Vector3d & center)
{
  local_pts_.clear();
  local_cols_.clear();
  if (!map_) {return;}
  const double R = params_.max_range + params_.local_map_margin;
  const double R2 = R * R;
  const Eigen::Vector3f c = center.cast<float>();
  for (const auto & p : map_->points) {
    const Eigen::Vector3f v(p.x, p.y, p.z);
    if (!v.allFinite()) {continue;}
    if ((v - c).squaredNorm() > R2) {continue;}
    local_pts_.push_back(v);
    local_cols_.emplace_back(p.b, p.g, p.r);
  }
}

void ImageMatcher::render(const Eigen::Isometry3d & T_map_cam, RenderResult & out) const
{
  const int W = size_.width;
  const int H = size_.height;
  out.bgr = cv::Mat::zeros(H, W, CV_8UC3);
  out.mask = cv::Mat::zeros(H, W, CV_8UC1);
  cv::Mat depth(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::infinity()));

  const Eigen::Isometry3d T_cam_map = T_map_cam.inverse();
  const Eigen::Matrix3f R = T_cam_map.rotation().cast<float>();
  const Eigen::Vector3f t = T_cam_map.translation().cast<float>();
  const float fx = static_cast<float>(fx_), fy = static_cast<float>(fy_);
  const float cx = static_cast<float>(cx_), cy = static_cast<float>(cy_);
  const float zmin = static_cast<float>(params_.min_range);
  const float zmax = static_cast<float>(params_.max_range);

  for (size_t i = 0; i < local_pts_.size(); ++i) {
    const Eigen::Vector3f pc = R * local_pts_[i] + t;
    const float z = pc.z();
    if (z < zmin || z > zmax) {continue;}
    const float inv_z = 1.0f / z;
    const int u = static_cast<int>(std::floor(fx * pc.x() * inv_z + cx));
    const int v = static_cast<int>(std::floor(fy * pc.y() * inv_z + cy));

    // 距離に応じたスプラット半径 (近い点ほど大きく描画して穴を埋める)
    int rad = static_cast<int>(0.5f * static_cast<float>(params_.point_size) * fx * inv_z + 0.5f);
    rad = std::clamp(rad, 0, params_.max_splat_px);

    for (int dv = -rad; dv <= rad; ++dv) {
      const int vv = v + dv;
      if (vv < 0 || vv >= H) {continue;}
      float * drow = depth.ptr<float>(vv);
      cv::Vec3b * crow = out.bgr.ptr<cv::Vec3b>(vv);
      uchar * mrow = out.mask.ptr<uchar>(vv);
      for (int du = -rad; du <= rad; ++du) {
        const int uu = u + du;
        if (uu < 0 || uu >= W) {continue;}
        if (z < drow[uu]) {  // Zバッファ
          drow[uu] = z;
          crow[uu] = local_cols_[i];
          mrow[uu] = 255;
        }
      }
    }
  }
  out.coverage = static_cast<double>(cv::countNonZero(out.mask)) / std::max(1, W * H);
}

void ImageMatcher::computeResiduals(
  const Eigen::Isometry3d & T_map_cam, std::vector<double> & residuals) const
{
  RenderResult rr;
  render(T_map_cam, rr);
  // レンダリングで点が存在する画素だけを実画像・仮想画像の両方で評価する
  // (点群が疎な領域を「背景色」として誤評価しないため)
  const ImageFeatures f_render = extractor_.extract(rr.bgr, rr.mask);
  const ImageFeatures f_target = extractor_.extract(target_, rr.mask);
  comparator_.compute(f_render, f_target, rr.coverage, residuals);
}

}  // namespace cdt_matching_ros2
