cmake_minimum_required(VERSION 3.0.2)
project(arrow_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  roscpp
  message_generation
)

add_message_files(
  FILES
  ArrowDetection2D.msg
  ArrowDetection3D.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/coordinates_2d.py
  scripts/coordinates_3d.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
