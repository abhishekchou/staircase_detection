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

#include <algorithm>
#include <vector>
#include <cmath>
#include <math.h>

//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
//typedef pcl::PointCloud<pcl::PointXYZRGB> colouredCloud;
 
//Bool Params
bool verbose = false;
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
 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr step_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud;
 
  double delta_angle;
  double z_passthrough;
  double distance_threshold;
  double cluster_tolerance;
  bool extract_bool;
  double step_width, step_depth;
  std::string input_cloud;
  std::string output_steps;
  std::string output_walls;
 
  void laserCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
  void preprocess_scene(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void removeOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  void findHorizontalPlanes();
  void removeWalls();
  void view_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  bool checkStep(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
	double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud);
  bool checkLength(std::vector<double> vertices);
  double computeDistance(std::vector<double> a, std::vector<double> b);

  void passThrough(double z);
 
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
  nh_private.param("step_width", step_width, 1.0);
  nh_private.param("step_depth", step_depth, 0.3);
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
  preprocess_scene(raw_cloud);
	//f new scene, will contain walls
	wall_removed = false;
}

//TODO change to function call by ref
/*
 @brief: Simple passthrough filter to test function operation on specific areas of the pointcloud map
 @param: PointCloud constptr
 */
void preprocess::passThrough(double z)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;

  pass.setInputCloud (step_cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (z-0.1, z+0.1);
  pass.filter (*cloud_filtered);
  step_cloud->swap(*cloud_filtered);
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
 @brief: Remove NANs followed by a Voxel Grid filter to reduce cloud size and resolution
 @param: PointCloud constptr
 */
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
/*
 @brief: Given the dimensions of the horizontal plane, check if valid step in order to localise
			 search to this area.
 @param: Dimensions of the step
 */
bool preprocess::checkStep(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{

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

  bool length = checkLength(coords);
  return length;
//  view_cloud(cloud);
  //compute shorter edge of the step
	//return true/false
}

/*
 @brief:
 @param:
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
  if(!verbose)
  {
    ROS_INFO("computed length of edge %f %f %f", edge[0], edge[1],edge[2]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_0[0],coord_0[1]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_1[0],coord_1[1]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_2[0],coord_2[1]);
    ROS_INFO("Coordinates of box (%f,%f)",coord_3[0],coord_3[1]);
  }


//  double x_length = std::abs(vertices[0]-vertices[2]);
//  double y_length = std::abs(vertices[1]-vertices[3]);

  if(edge[0] != 0)
  {
    if(edge[0] >= step_depth && edge[1] >= step_width)
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
 @brief: Cluster segmented planes, draw convex hulls around the clusters and ignore planes that are
			 not big/wide/deep enough to be stairs
 @param: constptr to current pointloud
 */
void preprocess::removeOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
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
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliers (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::ConcaveHull<pcl::PointXYZRGB> chull;
  //Convex Hull
  pcl::ConvexHull<pcl::PointXYZRGB> cx_hull;

  int j = 0;
  int step_count=0;
  std::vector<double> cluster_centers;
  if(cluster_indices.size() !=0)
  {

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      double avg_x, avg_y, avg_z;
      double counter;

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      {
        //copy clustered points into new pointcloud
        cloud_cluster->points.push_back (cloud->points[*pit]);
      }

      //createing concave hull around cluster
      //    chull.setInputCloud (cloud_cluster);
      //    chull.setAlpha(0.1);
      //    chull.reconstruct (*cloud_hull);
      //		pcl_pub.publish(cloud_hull);

      //creating convex hull around the cluster
      std::vector<pcl::Vertices> polygon;
      cx_hull.setInputCloud(cloud_cluster);
      cx_hull.setComputeAreaVolume(true);
      cx_hull.reconstruct(*cloud_hull,polygon);
      //    pcl_pub.publish(cloud_hull);
      //    view_cloud(cloud_hull);

      if(verbose)
      {
        if(cloud_hull->isOrganized()) ROS_INFO("HULL IS ORGANIZED");
        ROS_INFO("Cloud Hull Params-- height:%d width:%d", cloud_hull->height, cloud_hull->width);
        ROS_INFO("Cluster Individual Points %d", cloud_hull->points.at(1) );
      }


      if(checkStep(cloud_hull))
      {
        counter = 0;
        avg_x =0, avg_y=0; avg_z =0;
        //Compute mid point of the cluster
        pcl::PointCloud<pcl::PointXYZRGB>::const_iterator pit;
        for(pit = cloud_hull->begin(); pit != cloud_hull->end() ; ++pit)
        {
          avg_x += pit->x;
          avg_y += pit->y;
          avg_z += pit->z;
//          ROS_WARN("cluster points (%f,%f,%f)", pit->x, pit->y, pit->z);

          counter += 1;
        }
        avg_x = avg_x/counter;
        avg_y = avg_y/counter;
        avg_z = avg_z/counter;
        if(verbose) ROS_WARN("average points (%f,%f,%f) count %f", avg_x, avg_y, avg_z, counter);


        step_count++;
        steps->operator +=(*cloud_cluster);
        cluster_centers.push_back(avg_x);
        cluster_centers.push_back(avg_y);
        cluster_centers.push_back(avg_z);

        pcl::PointXYZRGB center;
        center.x = avg_x;
        center.y = avg_y;
        center.z = avg_z;
        center.r = 1.0;
        center.b = 0.0;
        center.g = 0.0;
        cloud_hull->points.push_back(center);
//        view_cloud(cloud_hull);
      }
      steps->width = steps->points.size ();
      steps->height = 1;
      steps->is_dense = true;
//      view_cloud(steps);
      pcl_pub.publish(steps);
//      ros::Duration(5).sleep();

      if(step_count == 2)
      {
        step_count = 0;
        if(verbose) ROS_WARN("cluster centers (%f,%f,%f) (%f,%f,%f)",cluster_centers[0], cluster_centers[1],
                                                   cluster_centers[2], cluster_centers[3],
                                                   cluster_centers[4], cluster_centers[5]);
        std::vector<double> a;
        std::vector<double> b;
        a.push_back(cluster_centers[0]);
        a.push_back(cluster_centers[1]);
        b.push_back(cluster_centers[3]);
        b.push_back(cluster_centers[4]);

        double dx = computeDistance(a,b);
        double dz = abs(cluster_centers[2]-cluster_centers[5]);
        //ROS_WARN("dx=%f dz=%f", dx, dz);

        double gradient = std::atan (dz/dx) * 180 / PI;
        if(gradient != 0.0) ROS_INFO("STAIR GRAIDENT IS %f", gradient);
//        ros::Duration(2).sleep();

      }
    }


    //  pcl_pub.publish(*outliers);
    //  ROS_INFO("cluster size %d", outliers->points.size());
    //  ROS_INFO("sleeping for 5 seconds now..yawwnnn");
    //  ros::Duration(5).sleep();
    return;
  }
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

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce (100);
    ros::Duration(.01).sleep();
  }
}

/*
 @brief: Main function doing all the fun stuff
 @param:
 */
void preprocess::findHorizontalPlanes()
{
  step_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  step_cloud->header.frame_id = raw_cloud->header.frame_id;
 
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  //Get cloud resolution to set RANSAC parameters
//  float resolution = computeCloudResolution(raw_cloud);
//  if(debug)ROS_INFO("Cloud Resolution: %f", resolution);

  //Segmentation parameters for planes
//  seg.setOptimizeCoefficients(true);
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

  int points_num = (int)raw_cloud->points.size();
  if(verbose) ROS_INFO("Staircase: Initial Cloud Size= %d", points_num);

  //Segmentation and plane extraction
  while(raw_cloud->points.size() > 0.01*points_num)
  {
    temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    seg.setInputCloud(raw_cloud);
    seg.segment(*inliers, *coefficients);
    if(verbose) ROS_INFO("Staircase: Segmented Plane - Inliers size =%d", (int) inliers->indices.size());
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
    if(temp_size <100000 && temp_size>100)
    {
      step_cloud->operator+=(*temp_cloud);
    }

    if(verbose) ROS_INFO("Staircase: Step Cloud Size = %d", step_cloud->points.size());
//    if(step_cloud->points.size() >0)
//      removeOutliers(step_cloud);


//    ROS_ERROR("Plane Coeff a=%f b=%f c=%f d=%f", coefficients->values[0],coefficients->values[1],coefficients->values[2],coefficients->values[3] );

    double counter = 0;
    double avg_z =0;
    pcl::PointCloud<pcl::PointXYZRGB>::const_iterator pit;
    for(pit = step_cloud->begin(); pit != step_cloud->end() ; ++pit)
    {
      avg_z += pit->z;
      counter += 1;
    }
    avg_z = avg_z / counter;
//    ROS_ERROR("Z average =%f",avg_z);

//    passThrough(avg_z);
    removeOutliers(step_cloud);
//    pcl_pub.publish(*step_cloud);
//    view_cloud(raw_cloud);

    //All points except those in the found plane
    temp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    temp_cloud->header.frame_id = raw_cloud->header.frame_id;
    extract.setInputCloud(raw_cloud);
    extract.setIndices(inliers);
    extract.setNegative(extract_bool);
    extract.filter(*temp_cloud);
    raw_cloud->swap(*temp_cloud);
    if(verbose) ROS_INFO("Staircase: Raw Cloud Size = %d Temp Cloud Size = %d", raw_cloud->points.size(), temp_cloud->points.size());


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
