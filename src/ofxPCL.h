#pragma once

// Eigen cause problems without these
#undef Success
#undef Status

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <pcl/search/pcl_search.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/registration/icp.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/surface/surfel_smoothing.h>
//#include <pcl/surface/marching_cubes_greedy.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/point_representation.h>

#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/registration/ia_ransac.h>

#include "typedefs.h"

#include "ofConstants.h"
#include "ofMesh.h"
#include "ofMatrix4x4.h"

//POINT_CLOUD_REGISTER_POINT_STRUCT (ofPoint,           // here we assume a XYZ + "test" (as fields)
//                                    (float, x, x)
//                                    (float, y, y)
//                                    (float, z, z))


namespace ofxPCL
{
void toOf(PointCloudPtr & cloud,
          ofMesh & mesh,
          float xfactor=1000,
          float yfactor=1000,
          float zfactor=1000);
void toOf(ColorPointCloudPtr & cloud,
          ofMesh & mesh,
          float xfactor=1000,
          float yfactor=1000,
          float zfactor=1000);
void toOf(ColorPointCloudPtr & cloud,
          SurfaceNormalsPtr & normals,
          ofMesh & mesh,
          float xfactor=1000,
          float yfactor=1000,
          float zfactor=1000);
void toOf(pcl::PointCloud<IntensityPointT>::Ptr & cloud,
          ofMesh & mesh,
          float xfactor=1000,
          float yfactor=1000,
          float zfactor=1000);
void toOf(pcl::PointCloud<IntensityPointNormalT>::Ptr & cloud,
          ofMesh & mesh,
          float xfactor=1000,
          float yfactor=1000,
          float zfactor=1000);

void toPCL(const ofMesh & mesh,
           PointCloudPtr pc);

void transform(PointCloudPtr cloud,
               ofMatrix4x4 matrix);

void smooth(ColorPointCloudPtr & cloud,
            ColorPointCloudPtr & mls_points,
            SurfaceNormalsPtr & mls_normals,
            float radius=0.03);

void getMesh(PointCloudPtr & cloud,
             ofMesh & mesh);

void marchingCubes(PointCloudPtr & cloud,
                   ofMesh & mesh);

void addIndices(ofMesh & mesh,
                MeshPtr & triangles);


SurfaceNormalsPtr calculateNormals(PointCloudPtr & cloud,
                                   float normal_radius_=0.02);

SurfaceNormalsPtr calculateNormalsOrdered(PointCloudPtr & cloud,
        pcl::IntegralImageNormalEstimation<PointT, NormalT>::NormalEstimationMethod normalEstimationMethod=pcl::IntegralImageNormalEstimation<PointT, NormalT>::AVERAGE_3D_GRADIENT,
        float maxDepthChangeFactor=0.02,
        float normalSmoothingSize=10);

SurfaceNormalsPtr calculateNormalsParallel(PointCloudPtr & cloud,
        float normal_radius_=0.02);


LocalDescriptorsPtr computeLocalFeatures (const PointCloudPtr & cloudIn,
        const SurfaceNormalsPtr & normals,
        float feature_radius=.1);

LocalDescriptorsPtr computeLocalFeatures (const PointNormalsPtr & cloudIn,
        float feature_radius=.1);
LocalDescriptorsPtr computeLocalFeaturesParallel (const PointNormalsPtr & cloudIn,
        float feature_radius=.1);

void voxelGridFilter(const PointCloudPtr & cloudIn,
                     PointCloud & cloudOut,
                     float leafSize=0.01);

void statisticalOutlierFilter(const PointCloudPtr & cloudIn,
                              PointCloud & cloudOut );

void removeDistanteFilter(const PointCloudPtr & cloudIn,
                          PointCloud & cloudOut ,
                          float depth_limit = 1.0);

MeshPtr gridProjection(PointCloudPtr & cloud,
                                     float res = 0.005,
                                     float padding = 3);

PointNormalsPtr movingLeastSquares(PointCloudPtr & cloud);

MeshPtr marchingCubes(PointCloudPtr & cloud,
                                    float leafSize);

MeshPtr greedyProjectionTriangulation(PointCloudPtr & cloud);


/* Use a PassThrough filter to remove points with depth values that are too large or too small */
PointCloudPtr thresholdDepth (const PointCloudPtr & input,
                              float min_depth,
                              float max_depth);

/* Use a VoxelGrid filter to reduce the number of points */
PointCloudPtr downsample (const PointCloudPtr & input,
                          float leaf_size);

/* Use a RadiusOutlierRemoval filter to remove all points with too few local neighbors */
PointCloudPtr removeOutliers (const PointCloudPtr & input,
                              float radius,
                              int min_neighbors);

void pair_align(const PointCloudPtr cloud_src,
                const PointCloudPtr cloud_tgt,
                ofMatrix4x4 &final_transform,
                bool downsample = true,
                double voxel_leaf_size = 0.005);

ofMatrix4x4 sampleConsensusInitialAlignment(PointNormalsPtr points_with_normals_src,
        PointNormalsPtr points_with_normals_tgt,
        LocalDescriptorsPtr source_features,
        LocalDescriptorsPtr target_features,
        double ia_min_sample_distance = 0.1,
        float ia_max_distance = 0.5,
        int ia_iterations = 100);
}
