cmake_minimum_required(VERSION 2.8.3)
project(staircase_detection)

set(PCL_DIR /usr/share/pcl-1.7)
set(OpenCV_DIR /usr/share/OpenCV)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_transport
  OpenCV
  PCL REQUIRED
  message_generation
)

add_message_files(
	FILES
	centroid_list.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  
#  INCLUDE_DIRS include
#  LIBRARIES 
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(preprocess src/preprocess.cpp)
#add_executable(validate_steps src/validate_steps.cpp)
target_link_libraries(preprocess  ${catkin_LIBRARIES})
add_dependencies(preprocess staircase_detection)
