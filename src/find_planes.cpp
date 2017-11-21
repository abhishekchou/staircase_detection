// Staircase detection module for Centauro Project.
// Find cascading planes in the filtered scene
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
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/common/io.h>

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

class plane_find
{
public:
	plane_find();
	~plane_find();
	ros::Subscriber scene;
	ros::Publisher planes;
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene;
	
	void sceneCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);

private:
	ros::NodeHandle nh;
	ros::NodeHandle nh_private;
	
};

plane_find::plane_find() : nh_private("~"){}

plane_find::~plane_find(){}

/*
 @brief: Callback function for the laser data
 @param: PointCloud ConstPtr
 */
void plane_find::sceneCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
	scene = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	scene = cloud;
}

/*
 @brief:
 @param:
 */
void plane_find::planeFind(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
	
}
