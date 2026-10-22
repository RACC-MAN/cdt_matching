# color_line_localization (ROS 2 Humble / C++)

色付き点群地図(PCD)から仮想カメラ画像をレンダリングし、実カメラ画像と
**色クラスごとの「近似線勾配・標準偏差・重心」** で比較して自己位置を推定するパッケージ。

## ファイル構成
| ファイル | 役割 |
|---|---|
| `include/.../image_matcher.hpp`, `src/image_matcher.cpp` | **画像マッチング** (ROS非依存): カメラモデル/歪み補正, 点群レンダリング, 色クラス特徴抽出, 残差計算 |
| `include/.../pose_optimizer.hpp`, `src/pose_optimizer.cpp` | **位置最適化**: `image_matcher.hpp` を include。数値ヤコビアン+LM法(NDTのパラメータ体系に準拠) |
| `include/.../localization_node.hpp`, `src/localization_node.cpp` | ROS2ノード: トピック/TF/初期位置/IMU・オドメトリ予測 |
| `config/params.yaml`, `launch/localization.launch.py` | 設定 / 起動 |

## 入出力
| 種別 | トピック | 型 |
|---|---|---|
| 入力 | `/input_img` | `sensor_msgs/msg/Image` |
| 入力(任意) | `/input_imu` | `sensor_msgs/msg/Imu` (`use_imu`) |
| 入力(任意) | `/input_odom` | `nav_msgs/msg/Odometry` (`use_odom`) ※`sensor_msgs/msg/odom` という型は存在しないため nav_msgs を使用 |
| 入力 | `/initialpose` | `geometry_msgs/msg/PoseWithCovarianceStamped` (RViz2 "2D Pose Estimate") |
| 出力 | `/cur_pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` |
| 出力 | `/map_cloud` | `sensor_msgs/msg/PointCloud2` (読み込んだ地図, transient_local) |
| 出力 | `/rendered_img` | `sensor_msgs/msg/Image` (推定姿勢でのレンダリング画像, デバッグ用) |
| TF | `map → odom` (`use_odom`時) / `map → base_link` (非使用時) | lidar_localization_ros2 と同じ方針 |

## ビルド・実行
```bash
cd ~/ros2_ws/src && cp -r color_line_localization .
cd ~/ros2_ws && rosdep install --from-paths src -y --ignore-src
colcon build --packages-select color_line_localization
source install/setup.bash
# config/params.yaml の camera_yaml_path / map_dir を編集してから
ros2 launch color_line_localization localization.launch.py
```
カメラYAMLは OpenCV 形式 (`camera_matrix`, `distortion_coefficients`, `image_width`, `image_height`) を想定。

## アルゴリズム概要
1. **前処理**: 入力画像を `image_scale` で縮小 → 歪み補正 (内部パラメータも同率で縮小)。
2. **レンダリング**: 推定開始位置の周辺点のみ抽出 → ピンホール投影 + Zバッファ + 距離依存スプラット。
3. **色分類**: B,G,R 各チャンネルを `color_levels` 階調に量子化し、(チャンネル,階調) ごとに画素をクラス分け。
   `color_levels: 8` → 3×8 = 24 クラス。
4. **特徴量(クラスごと)**: 占有率 / 重心(x,y) / 標準偏差(x,y) / 近似線勾配。
   近似線は画素座標の共分散行列の主軸(総最小二乗)。勾配は `elongation·(cos2θ, sin2θ)` で保持
   (垂直線の特異点・±90°の周期性を回避し、等方的なクラスでは勾配の影響が自動で小さくなる)。
5. **比較**: レンダリングで点が存在する画素のみを実画像側でも使用(疎な点群の空白を背景色と誤認しない)。
   両方に存在するクラスは占有率が大きいほど重みを大きくして残差化。片方にしかないクラスは占有率差のみがペナルティ。
6. **最適化**: 残差ベクトルに対する数値ヤコビアン + Levenberg–Marquardt。
   `max_iterations / transformation_epsilon / step_size` は NDT と同名。自由度は `optimize_dof_mask` で選択(既定: x, y, yaw)。
7. **予測**: オドメトリ使用時は差分で予測、IMU使用時はジャイロ積分で回転を置換。`score_threshold` 超過時は最適化結果を棄却し予測値を採用。

## 特徴量の追加方法
`ClassFeature` にメンバを追加 → `ColorFeatureExtractor::extract` で計算 → `FeatureComparator::compute`
で残差を追加 (`kResidualsPerClass` を更新) → `ImageMatcherParams` に重みを追加。最適化側の修正は不要です。

## 注意・制限 (重要)
* **本コードはビルド・実機検証をしていません。** 環境依存のコンパイルエラーが出る可能性があります。
* 評価値は画像間の色の違い(照明・露出・ホワイトバランス・カメラ個体差)に敏感です。
  地図作成時と実走行時で色味が違う場合、`color_levels` を粗くする、重みを調整する等が必要です。
* 初期位置から大きく離れると局所解に落ちます。RViz2 で初期位置を与えるか、オドメトリ/IMUで予測してください。
* 重みと `score_threshold` はデータ依存です。`ros2 run ... --ros-args --log-level debug` で score を見て調整してください。
* 加速度は並進予測に二重積分で使うとドリフトが大きいため、IMUは**ジャイロによる回転予測のみ**に使用しています。
* 地図は起動時に1回だけ読み込みます。点数が多い場合は `map_voxel_leaf` を大きくしてください。
* AtomS3R 側はカメラ画像を ROS2 側へ送る役割を想定しています(推定自体はPC/SBC上で実行)。
