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
 
 
//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
//typedef pcl::PointCloud<pcl::PointXYZRGB> colouredCloud;
 
//Bool Params
bool verbose = true;
bool debug = true;
bool wall_removed = false;
 
//PointCloud variables

#define PI 3.14
 
class preprocess
{
public:
  preprocess();
  ~preprocess();
  ros::Subscriber pcl_sub;
  ros::Publisher pcl_pub, wall_pub;
 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr step_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud;
 
  double delta_angle;
  double z_passthrough;
  double distance_threshold;
  double cluster_tolerance;
  bool extract_bool;
  std::string input_cloud;
  std::string output_steps;
  std::string output_walls;
 
  void laserCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
  void preprocess_scene(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void removeOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void findHorizontalPlanes();
  void removeWalls();
  void view_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  bool checkStep(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, double area);
	double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
	double computeLength(std::vector<pcl::Vertices> vertices, double area);

  void testing();
 
private:
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
 
};
 
preprocess::preprocess() : nh_private("~")
{
  //Load params form YAML input
  nh_private.getParam("input_cloud", input_cloud);
  nh_private.getParam("output_steps", output_steps);
  nh_private.getParam("output_walls", output_walls);
  nh_private.getParam("extract_bool", extract_bool);

  nh_private.param("delta_angle", delta_angle, 0.08);
  nh_private.param("cluster_tolerance", cluster_tolerance, 0.03);
  nh_private.param("z_passthrough", z_passthrough, 0.002);
//  nh_private.param("distance_threshold", distance_threshold);
 
  pcl_sub = nh.subscribe<sensor_msgs::PointCloud2>(input_cloud,
                                                   1000,
                                                   &preprocess::laserCallback,
                                                   this);
  pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(output_steps, 1000);
  wall_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(output_walls, 1000);
 
}
 
preprocess::~preprocess()
{}
 
void preprocess::laserCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  if(verbose)
  {
    ROS_INFO("Received new point cloud");
    wall_removed = false;
  }
  raw_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*msg, *raw_cloud);
  preprocess_scene(raw_cloud);
}
void preprocess::testing()
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;

  pass.setInputCloud (raw_cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (1.0f,3.0f);
  pass.filter (*cloud_filtered);
  raw_cloud->swap(*cloud_filtered);
}
void preprocess::removeWalls()
{
  //Remove Walls
  pcl::Normal _axis;
  _axis.normal_x = 0;
  _axis.normal_y = 0;
  _axis.normal_z = 1;

  //Set axis to compare surface normals to
  Eigen::Vector4f _ax = _axis.getNormalVector4fMap();

  //Passthrough filter to test operation of functions in specific areas of the pointcloud map
//  testing();

  //Estimate normals on all surfaces
  //TODO: Make faster by normal estimation only on RANSAC planes
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  ne.setInputCloud(raw_cloud);
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (0.06);
  ne.compute (*normals);

  size_t size = normals->points.size();
  pcl::PointCloud<pcl::PointXYZRGB>::iterator it = raw_cloud->begin();
  int count =0;
  for(size_t i=0; i< size ; ++i)
  {
    pcl::Normal norm = normals->points[i];
    Eigen::Vector4f norm_vec = norm.getNormalVector4fMap();
    double angle = pcl::getAngle3D(_ax ,norm_vec);
    if(angle > PI/4)
    {
      raw_cloud->erase(it);
    }
    it++;
  }

  wall_removed = true;
  return;
}
void preprocess::preprocess_scene(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr  &msg)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*msg, *cloud, indices);
  raw_cloud->swap(*cloud);

  //////////////////
  //Voxel Filtering
  //////////////////
  cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> vox;
  vox.setInputCloud (msg);
  float leaf = 0.04f;
  vox.setLeafSize(leaf,leaf,leaf);
  vox.filter(*cloud);
  raw_cloud->swap(*cloud);

  findHorizontalPlanes();
}
//Codeblock taken from pcl tutorials-correspondence grouping
//<http://pointclouds.org/documentation/tutorials/correspondence_grouping.php>
double preprocess::computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZRGB> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}
//TODO: Check if steps or false positive
bool preprocess::checkStep(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, double area)
{
	//is shorter edge of the rectangle deep enough?
	
	//return true/false
}

double preprocess::computeLength(std::vector<pcl::Vertices> vertices, double area)
{
	//draw bounding rectagle around convex hull
	
	//find longest edge of the rentangle
	
	//return longest edge
}
void preprocess::removeOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{

  // Creating the KdTree object for the search method
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud);

  //Euclidean clustering to identify false postives
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (cluster_tolerance); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (250000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud);
  ec.extract (cluster_indices);
  ROS_WARN("cluster indices size %d", cluster_indices.size());

  // Draw concave hull around the biggest plane found in the current scene
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliers (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ConcaveHull<pcl::PointXYZRGB> chull;
	//Convex Hull
	pcl::ConvexHull<pcl::PointXYZRGB> cx_hull;

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      //copy clustered points into new pointcloud
      cloud_cluster->points.push_back (cloud->points[*pit]);
    }

    //createing concave hull around cluster
    chull.setInputCloud (cloud_cluster);
    chull.setAlpha(0.1);
    chull.reconstruct (*cloud_hull);
		pcl_pub.publish(cloud_hull);
		
		//creating convex hull around the cluster
		std::vector<pcl::Vertices> vertices;
		cx_hull.setInputCloud(cloud_cluster);
		cx_hull.setComputeAreaVolume(true);
		cx_hull.reconstruct(*cloud_hull,vertices);
		
		

    if(verbose)
    {
      if(cloud_hull->isOrganized()) ROS_WARN("HULL IS ORGANIZED");
      ROS_WARN("CLOUD HULL PARAMS-- height:%d width:%d", cloud_hull->height, cloud_hull->width);
      ROS_WARN("cluster individual size %d", cloud_hull->points.size());
    }

    //Area of hull formed
    double area = 0;
		double length = computeLength(vertices,area);
    if(checkStep(cloud_cluster, area))
    {
       outliers->operator +=(*cloud_cluster);
    }
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    j++;
  }


  outliers->header.frame_id = raw_cloud->header.frame_id;
//  pcl_pub.publish(*outliers);
  ROS_WARN("cluster size %d", outliers->points.size());
  ROS_WARN("sleeping for 5 seconds now..yawwnnn");
  ros::Duration(5).sleep();
  return;
}
void preprocess::view_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
  pcl::visualization::PCLVisualizer viewer ("Output");
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//  cloud->header.frame_id = msg->header.frame_id;

//  pcl::copyPointCloud(msg,cloud);

//  cloud->resize(msg->height*msg->width);
//  for(size_t i = 0; i < cloud->points.size(); ++i)
//  {

//    ROS_WARN("Passes this-- %f", msg->points[i].x);
//    cloud->points.push_back(msg->points[i].x);
//    cloud->points.push_back(msg->points[i].y);
//    cloud->points.push_back(msg->points[i].z);
//  }

//  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//  ne.setInputCloud(cloud);
//  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
//  ne.setSearchMethod (tree);
//  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
//  ne.setRadiusSearch (0.6);
//  // Compute the features
//  ne.compute (*normals);

  int v1(0);
  viewer.createViewPort(0.0,0.0,1.0,1.0,v1);
  viewer.setBackgroundColor(0,0,0,v1);
  viewer.addText("Step Cloud", 10, 10, "window1",v1);
  viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "Step Cloud", v1);
//  viewer.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
//  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR,1.0,0,0, "normals");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Step Cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR,0,0,1.0, "Step Cloud");
  viewer.addCoordinateSystem (1.0);

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce (100);
    ros::Duration(.01).sleep();
  }
}
void preprocess::findHorizontalPlanes()
{
  step_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  step_cloud->header.frame_id = raw_cloud->header.frame_id;
 
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  //Get cloud resolution to set RANSAC parameters
//  float resolution = computeCloudResolution(raw_cloud);
//  if(debug)ROS_WARN("Cloud Resolution: %f", resolution);

  //Segmentation parameters for planes
//  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setAxis(Eigen::Vector3f(0,0,1));
  seg.setEpsAngle(delta_angle/2);
  seg.setDistanceThreshold(0.04);

  int points_num = (int)raw_cloud->points.size();
  if(verbose) ROS_WARN("Staircase: Initial Cloud Size= %d", points_num);

  //Segmentation and plane extraction
  while(raw_cloud->points.size() > 0.01*points_num)
  {
    temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    seg.setInputCloud(raw_cloud);
    seg.segment(*inliers, *coefficients);
    if(verbose) ROS_WARN("Staircase: Segmented Plane - Inliers size =%d", (int) inliers->indices.size());
    if(inliers->indices.size() ==0)
    {
      ROS_ERROR("STAITCASE_DETECTION :No inliers, could not find a plane perpendicular to Z-axis");
      break;
    }

    //Extract all points on the found plane (setNegative to false)
    extract.setInputCloud(raw_cloud);
    extract.setIndices(inliers);
    extract.setNegative(!extract_bool);
    extract.filter(*temp_cloud);
    size_t temp_size = temp_cloud->points.size();

    //If plane is too big -> floor/drivable, if plane too small -> outlier
    if(temp_size <300000 && temp_size>1000)
    {
      step_cloud->operator+=(*temp_cloud);
    }

    if(verbose) ROS_WARN("Staircase: Step Cloud Size = %d", step_cloud->points.size());
//    if(step_cloud->points.size() >0)
//      removeOutliers(step_cloud);

//    removeOutliers(step_cloud);
    pcl_pub.publish(*step_cloud);
//    view_cloud(raw_cloud);

//    ROS_WARN("sleeping for 5 seconds now..yawwnnn");
//    ros::Duration(1).sleep();

    //All points except those in the found plane
    temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    temp_cloud->header.frame_id = raw_cloud->header.frame_id;
    extract.setInputCloud(raw_cloud);
    extract.setIndices(inliers);
    extract.setNegative(extract_bool);
    extract.filter(*temp_cloud);
    raw_cloud->swap(*temp_cloud);
    if(verbose) ROS_WARN("Staircase: Raw Cloud Size = %d Temp Cloud Size = %d", raw_cloud->points.size(), temp_cloud->points.size());
    if(!wall_removed)
    {
      removeWalls();
    }

  }
//  pcl_pub.publish(*raw_cloud);

}
int main(int argc, char **argv)
{
  ros::init(argc, argv, "preprocess");
  ros::NodeHandle nh;
 
  preprocess pr;
 
  while(ros::ok())
    ros::spinOnce();
 
  return 0;
}
