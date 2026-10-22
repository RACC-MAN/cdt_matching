#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "cdt_matching_ros2/localization_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<cdt_matching_ros2::LocalizationNode>();
    rclcpp::spin(node);  // 単一スレッド実行: コールバック間の排他制御は不要
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("cdt_matching_ros2"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
