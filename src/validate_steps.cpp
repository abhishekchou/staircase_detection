// Staircase detection module for Centauro Project.
// Check area for planes that fit a staircase model
// Author: Abhishek Choudhary <acho@kth.se>

//_____ROS HEADERS____//
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Point.h>
#include <string.h>

//_____PCL HEADERS____//
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/Vertices.h>
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/common/io.h>
#include <pcl/common/centroid.h>

#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>

#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Geometry>

#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/surface/mls.h>
#include <pcl/console/parse.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/sample_consensus/sac_model_normal_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <staircase_detection/centroid_list.h>

#include <algorithm>
#include <vector>
#include <cmath>
#include <math.h>


void centroidCallback(staircase_detection::centroid_list::ConstPtr &msg)
{

  double theta = 0;
  Eigen::Affine3f rotate_z = Eigen::Affine3f::Identity();
  rotate_z.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
  pcl::transformPointCloud (*raw_cloud, *aligned_cloud, rotate_z);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "validate_steps");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("staircase_detection/steps_centroid", 1000, centroidCallback);

  ros::spin();

  return 0;
}
