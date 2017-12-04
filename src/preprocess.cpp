// Staircase detection module for Centauro Project.
// Using pointcloud data from a velodyne lidar
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

//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
//typedef pcl::PointCloud<pcl::PointXYZRGB> colouredCloud;
 
//Bool Params
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
  ros::Publisher pcl_pub, box_pub;
  ros::Publisher hypothesis_pub;
 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr step_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud;
 
  double delta_angle;
  double z_passthrough;
  double distance_threshold;
  double cluster_tolerance;
  double step_width, step_depth;
  double step_count;
  bool extract_bool, valid_step, verbose;

  std::string input_cloud;
  std::string output_steps;
  std::string step_maybe;
  std::string step_bounding_box;

  std::vector<double> centroid_x;
  std::vector<double> centroid_y;
  std::vector<double> centroid_z;

  void preprocess_scene();
  void removeWalls();
  void passThrough(double z);
  bool checkStep(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void view_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void laserCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
  void removeOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, double height);
  void findHorizontalPlanes(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);

  bool checkLength(std::vector<double> vertices);
  bool getStairParams(double height);

  double computeDistance(std::vector<double> a, std::vector<double> b);
  double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
 
private:
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
 
};
 
preprocess::preprocess() : nh_private("~")
{
  //Load params form YAML input
  nh_private.getParam("input_cloud", input_cloud);
  nh_private.getParam("output_steps", output_steps);
  nh_private.getParam("step_maybe", step_maybe);
  nh_private.getParam("step_bounding_box", step_bounding_box);
  nh_private.getParam("extract_bool", extract_bool);
  nh_private.getParam("verbose", verbose);

  nh_private.param("delta_angle", delta_angle, 0.08);
  nh_private.param("cluster_tolerance", cluster_tolerance, 0.03);
  nh_private.param("z_passthrough", z_passthrough, 0.002);
  nh_private.param("step_width", step_width, 1.0);
  nh_private.param("step_depth", step_depth, 0.25);
//  nh_private.param("distance_threshold", distance_threshold);
 
  pcl_sub = nh.subscribe<sensor_msgs::PointCloud2>(input_cloud,
                                                   1000,
                                                   &preprocess::laserCallback,
                                                   this);
  pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(output_steps, 1000);
  box_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(step_bounding_box, 1000);
  hypothesis_pub = nh.advertise<staircase_detection::centroid_list> (step_maybe, 1);
 
}
 
preprocess::~preprocess()
{}

/*
 @brief: Callback function for laser data
 @param: sensor_msgs::PointCloud2
 */
void preprocess::laserCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  if(verbose)
  {
    ROS_INFO("Received new point cloud");
    wall_removed = false;
  }
  raw_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*msg, *raw_cloud);
  step_count = 0;
  preprocess_scene();
  //if new scene, will contain walls
	wall_removed = false;
}

/*
 @brief: Remove NANs followed by a Voxel Grid filter to reduce cloud size and resolution
 @param: PointCloud constptr
 */
void preprocess::preprocess_scene()
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*raw_cloud, *cloud, indices);
  raw_cloud->swap(*cloud);

  //////////////////
  //Voxel Filtering
  //////////////////
  cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> vox;
  vox.setInputCloud (raw_cloud);
  float leaf = 0.04f;
  vox.setLeafSize(leaf,leaf,leaf);
  vox.filter(*cloud);
  raw_cloud->swap(*cloud);

  double z = -1.0;
  passThrough(z);
}

//TODO change to function call by ref
/*
 @brief: Simple passthrough filter to operate only on specific areas of the pointcloud map
 @param: filter from height z to z+0.5m
 */
void preprocess::passThrough(double z)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_copy (new pcl::PointCloud<pcl::PointXYZRGB>);
  plane_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  plane_cloud->header.frame_id = raw_cloud->header.frame_id;

  cloud_copy->operator +=(*raw_cloud);
  pass.setInputCloud (cloud_copy);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (z, z+0.3);
  pass.filter (*plane_cloud);
//  ROS_ERROR("Plane Cloud Deets %s", plane_cloud->header.frame_id.data());
//  view_cloud(plane_cloud);
//  plane_cloud->operator +=(*cloud_filtered);
  findHorizontalPlanes(plane_cloud);
}

/*
 @brief: Main function doing all the fun stuff
 @param:
 */
void preprocess::findHorizontalPlanes(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  //Get cloud resolution to set RANSAC parameters
//  float resolution = computeCloudResolution(raw_cloud);
//  if(debug)ROS_INFO("Cloud Resolution: %f", resolution);

  //Segmentation parameters for planes
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setAxis(Eigen::Vector3f(0,0,1));
  seg.setEpsAngle(delta_angle/2);
  seg.setDistanceThreshold(0.04);

//  if(!wall_removed)
//  {
//    removeWalls();
//  }

  int points_num = (int)plane_cloud->points.size();
  if(verbose) ROS_INFO("Staircase: Initial Cloud Size= %d", points_num);

  //Segmentation and plane extraction
  while(plane_cloud->points.size() > 0.01*points_num)
  {
    temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    step_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    step_cloud->header.frame_id = raw_cloud->header.frame_id;

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    seg.setInputCloud(plane_cloud);
    seg.segment(*inliers, *coefficients);
    if(verbose) ROS_INFO("Staircase: Segmented Plane - Inliers size =%d", (int) inliers->indices.size());
    if(inliers->indices.size() ==0)
    {
      ROS_ERROR("STAITCASE_DETECTION :No inliers, could not find a plane perpendicular to Z-axis");
      break;
    }

    //Extract all points on the found plane (setNegative to false)
    extract.setInputCloud(plane_cloud);
    extract.setIndices(inliers);
    extract.setNegative(!extract_bool);
    extract.filter(*temp_cloud);
    size_t temp_size = temp_cloud->points.size();

    //Find z-intercept of plane (ax+by+cz+d=0 => z = -d/c)
    double c = coefficients->values[2];
    double d = coefficients->values[3];
    double height = (-d/c);

    //If plane is too big -> floor/drivable, if plane too small -> outlier
    if(temp_size <50000 && temp_size>500)
    {
      step_cloud->operator+=(*temp_cloud);
    }

    if(verbose)
      ROS_INFO("Staircase: Planes found at height = %f", height);
    pcl_pub.publish(step_cloud);

    if(step_cloud->points.size() !=0)
      removeOutliers(step_cloud, height);
    else
      passThrough(height+0.1);

//    ROS_ERROR("Plane Coeff a=%f b=%f c=%f d=%f", coefficients->values[0],coefficients->values[1],coefficients->values[2],coefficients->values[3] );
  }
}

/*
 @brief: Cluster segmented planes, draw convex hulls around the clusters and ignore planes that are
       not big/wide/deep enough to be stairs
 @param: constptr to current pointloud
 */

void preprocess::removeOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, double height)
{

  // Creating the KdTree object for the search method
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr steps (new pcl::PointCloud<pcl::PointXYZRGB>);
  steps->header.frame_id = raw_cloud->header.frame_id;

  //Euclidean clustering to identify false postives
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (cluster_tolerance); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (250000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud);
  ec.extract (cluster_indices);
  if(verbose) ROS_INFO("cluster indices size %d", cluster_indices.size());

  // Draw concave hull around the biggest plane found in the current scene
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZRGB>);
  cloud_hull->header.frame_id = steps->header.frame_id;

  //Convex Hull
  pcl::ConvexHull<pcl::PointXYZRGB> cx_hull;

  if(cluster_indices.size() !=0)
  {

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      cloud_cluster->header.frame_id = raw_cloud->header.frame_id;
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      {
        //cloud_hull clustered points into new pointcloud
        cloud_cluster->points.push_back (cloud->points[*pit]);
      }
//      ROS_WARN("Cluster size %d", it->indices.size());

      //creating convex hull around the cluster
      std::vector<pcl::Vertices> polygon;
      cx_hull.setInputCloud(cloud_cluster);
      cx_hull.setComputeAreaVolume(true);
      cx_hull.reconstruct(*cloud_hull,polygon);

      if(checkStep(cloud_hull))
      {
//        ROS_ERROR("Step found at height= %f", height);
        step_count ++;
        steps->operator +=(*cloud_cluster);
        box_pub.publish(steps);
//        ros::Duration(2).sleep();
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_cluster, centroid);
        centroid_x.push_back(centroid[0]);
        centroid_y.push_back(centroid[1]);
        centroid_z.push_back(centroid[2]);
//        ROS_WARN("z centroid %f", centroid_z.back());

        //assumption that there is only one staircase in the scene
        //so there will be only valid step at a particular height
        break;
      }
    }
    if(step_count >= 2)
    {
      bool eureka = getStairParams(height);
//      if(eureka)
//        box_pub.publish(steps);
    }
  }
  passThrough(height);
  return;
}


//TODO: Check if steps or false positive
/*
 @brief: Given the dimensions of the horizontal plane, check if valid step in order to localise
       search to this area.
 @param: Dimensions of the step
 */
bool preprocess::checkStep(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
  valid_step = false;
//  size_t hull_size = cloud->points.size();
  pcl::PointCloud<pcl::PointXYZRGB>::const_iterator it= cloud->begin();
  double max_x = -50, y_max_x= -50;
  double max_y = -50, x_max_y= -50;
  double min_x = 50, y_min_x= 50;
  double min_y = 50, x_min_y= 50;

  if(verbose) ROS_ERROR("cluster size %d",cloud->width);

  for(it = cloud->begin(); it != cloud->end(); it++)
  {
//    ROS_INFO("min_x, current x %f %f",min_x, it->x);
    if(min_x > it->x)
    {
//      ROS_INFO("min_x new %f",min_x);
      min_x = it->x;
      y_min_x = it->y;
    }
  }
  for(it = cloud->begin(); it != cloud->end(); it++)
  {
    if(min_y > it->y)
    {
      min_y = it->y;
      x_min_y = it->x;
    }
  }

  for(it= cloud->begin(); it != cloud->end(); it++)
  {
    if(max_x < it->x)
    {
      max_x = it->x;
      y_max_x = it->y;
    }
  }
  for(it = cloud->begin(); it != cloud->end(); it++)
  {
    if(max_y < it->y)
    {
      max_y = it->y;
      x_max_y = it->x;
    }
  }


//  ROS_INFO("Min Pair (%f,%f)",min_x, min_y);
//  ROS_INFO("Max Pair (%f,%f)",max_x, max_y);

  std::vector<double> coords;
  coords.push_back(min_x);
  coords.push_back(y_min_x);
  coords.push_back(x_min_y);
  coords.push_back(min_y);
  coords.push_back(max_x);
  coords.push_back(y_max_x);
  coords.push_back(x_max_y);
  coords.push_back(max_y);
  if(verbose)
  {
//  ROS_INFO("Intended Coordinates of box (%f,%f)",coords[0],coords[1]);
  ROS_INFO("Intended Coordinates(%f,%f)",min_x,y_min_x);
//  ROS_INFO("Intended Coordinates of box (%f,%f)",coords[2],coords[3]);
  ROS_INFO("Intended Coordinates(%f,%f)",x_min_y,min_y);
//  ROS_INFO("Intended Coordinates of box (%f,%f)",coords[4],coords[5]);
  ROS_INFO("Intended Coordinates(%f,%f)",max_x,y_max_x);
//  ROS_INFO("Intended Coordinates of box (%f,%f)",coords[6],coords[7]);
  ROS_INFO("Intended Coordinates(%f,%f)",x_max_y,max_y);
  }

  valid_step = checkLength(coords);
  return valid_step;
}

/*
 @brief: compute euclidean distsance between two points
 @param: vector<double> start and end points
 */
double preprocess::computeDistance(std::vector<double> a, std::vector<double> b)
{
  double x1,y1,x2,y2;
  x1 = a[0];
  x2 = b[0];
  y1 = a[1];
  y2 = b[1];
  double length = std::sqrt (std::pow((x1-x2),2) + std::pow((y1-y2),2) );
  return length;
}

/*
 @brief: Given the convex hull around bottom step, compute the length and equation of the longest
       edge of the step.
 @param: Vertices of the convex polygon, area od the polygon
 */
bool preprocess::checkLength(std::vector<double> vertices)
{
  double edge[6];
  bool result = false;

  std::vector<double> coord_0;
  std::vector<double> coord_1;
  std::vector<double> coord_2;
  std::vector<double> coord_3;
  coord_0.push_back(vertices[0]);
  coord_0.push_back(vertices[1]);
  coord_1.push_back(vertices[2]);
  coord_1.push_back(vertices[3]);
  coord_2.push_back(vertices[4]);
  coord_2.push_back(vertices[5]);
  coord_3.push_back(vertices[6]);
  coord_3.push_back(vertices[7]);

  edge[0] = computeDistance(coord_0, coord_1);
  edge[1] = computeDistance(coord_0, coord_2);
  edge[2] = computeDistance(coord_0, coord_3);
  edge[3] = computeDistance(coord_1, coord_2);
  edge[4] = computeDistance(coord_1, coord_3);
  edge[5] = computeDistance(coord_2, coord_3);

  std::sort(edge, edge+6);
  if(verbose)
  {
    ROS_INFO("computed length of edge %f %f %f %f %f %f", edge[0], edge[1], edge[2], edge[3], edge[4], edge[5]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_0[0],coord_0[1]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_1[0],coord_1[1]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_2[0],coord_2[1]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_3[0],coord_3[1]);
  }

//  double x_length = std::abs(vertices[0]-vertices[2]);
//  double y_length = std::abs(vertices[1]-vertices[3]);

  if(edge[0] != 0)
  {
    if(edge[0] >= step_depth && edge[2] >= step_width)
    {
      result = true;
      ROS_ERROR("This is okay");
    }
  }

  if(result == false)
    ROS_ERROR("Not an okay step!");
  return result;
}

/*
 @brief:
 @param:
*/
bool preprocess::getStairParams(double height)
{
  std::vector<double> a, b;
  a.push_back(centroid_x.at(0));
  b.push_back(centroid_x.at(1));
  a.push_back(centroid_y.at(0));
  b.push_back(centroid_y.at(1));

  geometry_msgs::Point centroid;
  staircase_detection::centroid_list msg;
  centroid.x = a[0]; centroid.y = a[1];
  msg.centroids.push_back(centroid);
  centroid.x = b[0]; centroid.y = b[1];
  msg.centroids.push_back(centroid);

  double w = computeDistance(a,b);
  double h = std::abs(centroid_z[0] - centroid_z[1]);
  //Check if step depth is greater than threshold and less than
  //1m (becomes drivable) and height is less that 25cm and greater than
  //5cm
  if(w>=step_depth && w<=1.0 && h<= 0.25 && h>=0.05)
  {
    ROS_ERROR("EUREKA! %fcm behind and %fcm above", w*100, h*100);
    return true;
    hypothesis_pub.publish(msg);
  }
  else
  {
    ROS_ERROR("Something is not right :( w=%f h=%f",  w*100, h*100);
    return false;
  }
}

//TODO change to function call by reference
/*
 @brief: Compare surface normals with user-defined axis (z-axis), remove points from cloud that are
       not parallel. Filter out walls, steeps ramps etc from the scene.
 @param: PointCloud constptr
 */
void preprocess::removeWalls()
{
  //Define Axis perpendicular to normals on walls/surfaces to filter out
  pcl::Normal _axis;
  _axis.normal_x = 0;
  _axis.normal_y = 0;
  _axis.normal_z = 1;
  Eigen::Vector4f _ax = _axis.getNormalVector4fMap();

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
  //iterate through all estimated normals
  for(size_t i=0; i< size ; ++i)
  {
    pcl::Normal norm = normals->points[i];
    Eigen::Vector4f norm_vec = norm.getNormalVector4fMap();
    //angle between surface normals
    double angle = pcl::getAngle3D(_ax ,norm_vec);
    if(angle > PI/4) //more than 45deg angle made with z-axis, definitely not horizontal plane
    {
      raw_cloud->erase(it);
    }
    it++;
  }

  wall_removed = true;
  return;
}

/*
 @brief: Codeblock taken from pcl tutorials-correspondence grouping.
			 <http://pointclouds.org/documentation/tutorials/correspondence_grouping.php>
 @param: PointCloud constptr
 */
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
      res += std::sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

/*
 @brief: PCL Visualizer function to view pointcloud outside RVIZ
 @param: constptr to pointloud
 */
void preprocess::view_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
  pcl::visualization::PCLVisualizer viewer ("Output");
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//  cloud->header.frame_id = msg->header.frame_id;

//  pcl::copyPointCloud(msg,cloud);

//  cloud->resize(msg->height*msg->width);
//  for(size_t i = 0; i < cloud->points.size(); ++i)
//  {

//    ROS_INFO("Passes this-- %f", msg->points[i].x);
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
//  viewer.spinOnce (100);

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
    ros::Duration(.01).sleep();
  }
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
