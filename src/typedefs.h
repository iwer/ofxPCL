#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

/* Define some custom types to make the rest of our code easier to read */

namespace ofxPCL
{

// Define "PointCloud" to be a pcl::PointCloud of pcl::PointXYZ points
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;

// Define "ColorPointCloud" to be a pcl::PointCloud of pcl::PointXYZRGB points
typedef pcl::PointXYZRGB ColorPointT;
typedef pcl::PointCloud<ColorPointT> ColorPointCloud;
typedef pcl::PointCloud<ColorPointT>::Ptr ColorPointCloudPtr;
typedef pcl::PointCloud<ColorPointT>::ConstPtr ColorPointCloudConstPtr;

// Define "IntensityPointCloud" to be a pcl::PointCloud of pcl::PointXYZI points
typedef pcl::PointXYZI IntensityPointT;
typedef pcl::PointCloud<IntensityPointT> IntensityPointCloud;
typedef pcl::PointCloud<IntensityPointT>::Ptr IntensityPointCloudPtr;
typedef pcl::PointCloud<IntensityPointT>::ConstPtr IntensityPointCloudConstPtr;

// Define "SurfaceNormals" to be a pcl::PointCloud of pcl::Normal points (only normal)
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<NormalT> SurfaceNormals;
typedef pcl::PointCloud<NormalT>::Ptr SurfaceNormalsPtr;
typedef pcl::PointCloud<NormalT>::ConstPtr SurfaceNormalsConstPtr;

// Define "PointNormals" to be a pcl::PointCloud of pcl::PointNormal points (point and normal)
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointNormals;
typedef pcl::PointCloud<PointNormalT>::Ptr PointNormalsPtr;
typedef pcl::PointCloud<PointNormalT>::ConstPtr PointNormalsConstPtr;

// Define "IntensityPointNormals" to be a pcl::PointCloud of pcl::PointXYZINormal points
typedef pcl::PointXYZINormal IntensityPointNormalT;
typedef pcl::PointCloud<IntensityPointNormalT> IntensityPointNormals;
typedef pcl::PointCloud<IntensityPointNormalT>::Ptr IntensityPointNormalsPtr;
typedef pcl::PointCloud<IntensityPointNormalT>::ConstPtr IntensityPointNormalsConstPtr;

// Define "LocalDescriptors" to be a pcl::PointCloud of pcl::FPFHSignature33 points
typedef pcl::FPFHSignature33 LocalDescriptorT;
typedef pcl::PointCloud<LocalDescriptorT> LocalDescriptors;
typedef pcl::PointCloud<LocalDescriptorT>::Ptr LocalDescriptorsPtr;
typedef pcl::PointCloud<LocalDescriptorT>::ConstPtr LocalDescriptorsConstPtr;

// Define "GlobalDescriptors" to be a pcl::PointCloud of pcl::VFHSignature308 points
typedef pcl::VFHSignature308 GlobalDescriptorT;
typedef pcl::PointCloud<GlobalDescriptorT> GlobalDescriptors;
typedef pcl::PointCloud<GlobalDescriptorT>::Ptr GlobalDescriptorsPtr;
typedef pcl::PointCloud<GlobalDescriptorT>::ConstPtr GlobalDescriptorsConstPtr;

typedef pcl::PolygonMesh Mesh;
typedef pcl::PolygonMeshPtr MeshPtr;
typedef pcl::PolygonMeshConstPtr MeshConstPtr;
}
#endif
