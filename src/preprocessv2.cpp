// Preprocess the velodyne pointcloud
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
 
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
 
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
 
#include <pcl/segmentation/sac_segmentation.h>
 
 
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
//typedef pcl::PointCloud<pcl::PointXYZRGB> colouredCloud;
 
//Bool Params
bool verbose = false;
 
//PointCloud variables
 
class preprocess
{
public:
  preprocess();
  ~preprocess();
  ros::Subscriber pcl_sub;
  ros::Publisher pcl_pub;
 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr step_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud;
 
  double delta_angle;
  bool extract_bool;
  std::string input_cloud;
  std::string output_steps;
 
  void velodyneCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
  void preprocess_scene(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &msg);
  void findHorizontalPlanes();
 
private:
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
 
};
 
preprocess::preprocess() : nh_private("~")
{
  //Load params form YAML input
  nh_private.getParam("input_cloud", input_cloud);
  nh_private.getParam("output_steps", output_steps);
  nh_private.getParam("extract_bool", extract_bool);
  nh_private.param("delta_angle", delta_angle, 0.08);
 
  pcl_sub = nh.subscribe<sensor_msgs::PointCloud2>(input_cloud,
                                                   1000,
                                                   &preprocess::velodyneCallback,
                                                   this);
  pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(output_steps, 1000);
 
}
 
preprocess::~preprocess()
{}
 
void preprocess::velodyneCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  if(verbose) ROS_INFO("Receive new point cloud");
  raw_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*msg, *raw_cloud);
  preprocess_scene(raw_cloud);
}
 
void preprocess::preprocess_scene(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr  &msg)
{
  //////////////////
  //Voxel Filtering
  //////////////////
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> vox;
  vox.setInputCloud (msg);
  float leaf = 0.04f;
  vox.setLeafSize(leaf,leaf,leaf);
  vox.filter(*cloud);
  raw_cloud.swap(cloud);
  findHorizontalPlanes();
  
}
 
void preprocess::findHorizontalPlanes()
{
  step_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  step_cloud->header.frame_id = raw_cloud->header.frame_id;
 
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
 
  //Segment plane perpendicular to Z axis
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setAxis(Eigen::Vector3f(0,0,1));
  seg.setEpsAngle(delta_angle/2);
  seg.setDistanceThreshold(0.1);
 

  int points_num = (int)raw_cloud->points.size();
  while(raw_cloud->points.size() > 0.1*points_num)
  {
    temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    seg.setInputCloud(raw_cloud);
    seg.segment(*inliers, *coefficients);
 
    if(inliers->indices.size() ==0)
    {
      if(verbose) ROS_ERROR("STAITCASE_DETECTION :No inliers, could not find a plane perpendicular to Z-axis");
    }

//    pcl::copyPointCloud(raw_cloud, inliers, step_cloud);

    //All points except found plane
    extract.setNegative(extract_bool);
    extract.filter(*temp_cloud);
    raw_cloud->swap(*temp_cloud);

 
    // extract.setInputCloud(raw_cloud);
    // extract.setIndices(inliers);
    
    // //All points except found plane
    // extract.setNegative(extract_bool);
    // extract.filter(*temp_cloud);
    // raw_cloud->swap(*temp_cloud);
    
    // //All points on found plane
    // coefficients = pcl::ModelCoefficients::Ptr (new pcl::ModelCoefficients);
    // inliers = pcl::PointIndices::Ptr (new pcl::PointIndices);
    // seg.setInputCloud(raw_cloud);
    // seg.segment(*inliers, *coefficients);
    
    // ROS_WARN("here now");
    // temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    // extract.setNegative(!extract_bool);
    // extract.filter(*temp_cloud);
    // step_cloud->operator+=(*temp_cloud);

    pcl_pub.publish(*step_cloud);



    // step_cloud->operator +(*plane_seg);
    
    // extract.filter(*plane_seg);
    // raw_cloud->swap(*plane_seg);
    
    // ROS_INFO("_staircase detection: Cloud Size = %d",raw_cloud->points.size());
 
  }
  // pcl_pub.publish(*step_cloud);
 
 // pcl::visualization::PCLVisualizer viewer ("Output");
 
 // int v1(0);
 // viewer.createViewPort(0.0,0.0,0.3,0.3,v1);
 // viewer.setBackgroundColor(0,0,0,v1);
 // viewer.addText("Original Scan", 10, 10, "window1",v1);
 // // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(raw_cloud);
 // viewer.addPointCloud<pcl::PointXYZRGB>(raw_cloud, "Original Scene", v1);
 
 // int v2(1);
 // viewer.createViewPort(0.3,0.3,0.6,0.6, v2);
 // viewer.setBackgroundColor(0,0,0,v2);
 // viewer.addText("Step Plane", 10, 10, "window2",v2);
 // viewer.addPointCloud<pcl::PointXYZRGB>(step_cloud, "Step Cloud", v2);


 // int v3(2);
 // viewer.createViewPort(0.6, 0.6, 1.0, 1.0, v3);
 // viewer.setBackgroundColor(0,0,0,v3);
 // viewer.addText("Plane Seg", 10, 10, "window3",v3);
 // viewer.addPointCloud<pcl::PointXYZRGB>(plane_seg,"Plane Seg", v3);
  
 // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Scan");
 // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Step Plane");
 // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Plane Seg");
 // viewer.addCoordinateSystem (1.0);
 
 // while (!viewer.wasStopped ())
 // {
 //   viewer.spinOnce (100);
 //   ros::Duration(.01).sleep();
 // }
 
}
 
int main(int argc, char **argv)
{
  ros::init(argc, argv, "preprocess");
  ros::NodeHandle nh;
 
  preprocess pr;
 
  while(ros::ok())
  {
    ros::spinOnce();
  }
 
  return 0;
}
 
