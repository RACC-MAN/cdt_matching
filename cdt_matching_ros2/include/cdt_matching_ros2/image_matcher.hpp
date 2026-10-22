#ifndef CDT_MATCHING_ROS2__IMAGE_MATCHER_HPP_
#define CDT_MATCHING_ROS2__IMAGE_MATCHER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>
#include <vector>

namespace cdt_matching_ros2
{

// ---------------------------------------------------------------------------
// パラメータ
// ---------------------------------------------------------------------------
struct ImageMatcherParams
{
  // --- 色分類 ---
  // 各チャンネル(B,G,R)を color_levels 階調に量子化し、(チャンネル, 階調) ごとにクラスを作る。
  // 例: color_levels = 8 → 3ch × 8 = 24 クラス
  int color_levels = 8;
  int min_class_pixels = 10;  // これ未満のピクセル数のクラスは無効扱い

  // --- 残差の重み (特徴量を追加したらここに重みを足す) ---
  double weight_ratio = 1.0;     // クラス占有率
  double weight_centroid = 1.0;  // 重心座標
  double weight_stddev = 1.0;    // 標準偏差
  double weight_gradient = 1.0;  // 近似線の勾配(ベクトル表現)
  double weight_coverage = 0.5;  // 点群が写っている画素の割合
  double min_coverage = 0.02;    // これ未満なら empty_render_penalty を加算
  double empty_render_penalty = 1.0;

  // --- レンダリング ---
  double min_range = 0.1;        // [m]
  double max_range = 15.0;       // [m]
  double point_size = 0.03;      // [m] 点の実寸(スプラット半径の算出に使用)
  int max_splat_px = 3;          // スプラット半径の上限 [px]
  double local_map_margin = 2.0; // ローカルマップ抽出時の余裕 [m]
};

// ---------------------------------------------------------------------------
// カメラモデル (OpenCV キャリブレーション YAML を読み込む)
// ---------------------------------------------------------------------------
class CameraModel
{
public:
  // YAML 内のキー: camera_matrix, distortion_coefficients, image_width, image_height
  bool loadFromYaml(const std::string & path, std::string & error);

  // 入力画像サイズと縮小率から、処理用の内部パラメータと歪み補正マップを作る
  void configure(const cv::Size & input_size, double scale);

  // 縮小 + 歪み補正 (入力: bgr8, 出力: 処理解像度の bgr8)
  cv::Mat preprocess(const cv::Mat & bgr) const;

  bool isLoaded() const {return loaded_;}
  bool isConfigured() const {return configured_;}
  const cv::Size & procSize() const {return proc_size_;}
  double fx() const {return fx_;}
  double fy() const {return fy_;}
  double cx() const {return cx_;}
  double cy() const {return cy_;}

private:
  cv::Mat K_calib_;
  cv::Mat dist_;
  cv::Size calib_size_{0, 0};
  cv::Size proc_size_{0, 0};
  cv::Mat map1_, map2_;
  double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
  bool loaded_ = false;
  bool configured_ = false;
};

// ---------------------------------------------------------------------------
// 色クラスごとの特徴量
// ---------------------------------------------------------------------------
struct ClassFeature
{
  bool valid = false;
  int count = 0;
  double ratio = 0.0;                                  // 占有率 (対象画素数に対する割合)
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();  // 重心 (max(w,h)で正規化)
  Eigen::Vector2d stddev = Eigen::Vector2d::Zero();    // x,y標準偏差 (同上)
  double slope_angle = 0.0;   // 近似直線(主軸)の角度 [rad], (-pi/2, pi/2]  ※デバッグ表示用
  double elongation = 0.0;    // 0:等方 ～ 1:完全な線状
  // 最適化用の連続表現: elongation * (cos2θ, sin2θ)
  // 角度の周期性(±π/2)や垂直線での特異点を避けつつ、等方なクラスの角度ノイズを抑える
  Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
};

struct ImageFeatures
{
  int levels = 0;
  int total_pixels = 0;
  std::vector<ClassFeature> classes;  // index = channel * levels + level (channel 0=B,1=G,2=R)
};

class ColorFeatureExtractor
{
public:
  ColorFeatureExtractor(int levels, int min_pixels);
  // mask が空なら全画素、非空なら mask != 0 の画素のみ使用
  ImageFeatures extract(const cv::Mat & bgr, const cv::Mat & mask) const;
  int levels() const {return levels_;}

private:
  int levels_;
  int min_pixels_;
};

// ---------------------------------------------------------------------------
// 特徴量の比較 → 残差ベクトル
// ---------------------------------------------------------------------------
class FeatureComparator
{
public:
  // 1クラスあたりの残差数: ratio(1) + centroid(2) + stddev(2) + gradient(2)
  static constexpr int kResidualsPerClass = 7;

  explicit FeatureComparator(const ImageMatcherParams & params) : params_(params) {}

  // 残差の並び: [クラス0の7個, クラス1の7個, ..., coverage(1個)]
  void compute(
    const ImageFeatures & rendered, const ImageFeatures & target,
    double coverage, std::vector<double> & residuals) const;

  static size_t residualSize(int levels) {return 3 * levels * kResidualsPerClass + 1;}

private:
  ImageMatcherParams params_;
};

// ---------------------------------------------------------------------------
// 画像マッチング本体: 点群レンダリング + 特徴量比較
// ---------------------------------------------------------------------------
struct RenderResult
{
  cv::Mat bgr;   // CV_8UC3
  cv::Mat mask;  // CV_8UC1 (点が描画された画素=255)
  double coverage = 0.0;
};

class ImageMatcher
{
public:
  using Cloud = pcl::PointCloud<pcl::PointXYZRGB>;

  explicit ImageMatcher(const ImageMatcherParams & params);

  void setCameraIntrinsics(double fx, double fy, double cx, double cy, const cv::Size & size);
  void setMap(const Cloud::ConstPtr & map);
  // 前処理済み(縮小+歪み補正済み)の実画像をセット
  void setTargetImage(const cv::Mat & bgr);
  // 推定開始位置の周辺点だけを抽出 (1回の最適化につき1回呼ぶ)
  void prepareLocalMap(const Eigen::Vector3d & center);

  // 仮想カメラ(T_map_cam: カメラ光学座標系 z前/x右/y下 の map 上の姿勢)でレンダリング
  void render(const Eigen::Isometry3d & T_map_cam, RenderResult & out) const;

  // レンダリング→特徴抽出→比較して残差を返す
  void computeResiduals(const Eigen::Isometry3d & T_map_cam, std::vector<double> & residuals) const;

  const ImageMatcherParams & params() const {return params_;}

private:
  ImageMatcherParams params_;
  ColorFeatureExtractor extractor_;
  FeatureComparator comparator_;

  Cloud::ConstPtr map_;
  std::vector<Eigen::Vector3f> local_pts_;
  std::vector<cv::Vec3b> local_cols_;  // BGR

  cv::Mat target_;
  cv::Size size_{0, 0};
  double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
};

}  // namespace cdt_matching_ros2

#endif  // cdt_matching_ros2__IMAGE_MATCHER_HPP_
