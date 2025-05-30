cmake_minimum_required(VERSION 3.8)
project(ariaNode)

include_directories(/usr/local/include/AriaCoda)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

message("Include directories: /usr/local/include/AriaCoda")
message("The root directory is ${CMAKE_MODULE_PATH}")

#find_library(ARIA_LIB Aria)

#if(ARIA_LIB)
#  message("the Aria library was found: ${ARIA_LIB}")
#endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(ariaNode src/ariaNode.cpp)
#ament_target_dependencies(ariaNode
#"geometry_msgs"
#"rclcpp"
#)

ament_target_dependencies(ariaNode geometry_msgs rclcpp tf2 tf2_geometry_msgs nav_msgs)

#target_link_libraries(
#  ariaNode
#  libAria.so
#)

target_link_libraries(ariaNode libAria.so)

install(TARGETS
ariaNode
DESTINATION lib/${PROJECT_NAME})

ament_package()
