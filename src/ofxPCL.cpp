#include "ofxPCL.h"

namespace ofxPCL
{
void toOf(ColorPointCloudPtr & cloud, ofMesh & mesh, float xfactor, float yfactor, float zfactor)
{
    mesh.setMode(OF_PRIMITIVE_POINTS);
    mesh.getVertices().resize(cloud->points.size());
    mesh.getColors().resize(cloud->points.size());
    int i=0, n=0;
    if(cloud->is_dense)
    {
        for(int i=0; i<cloud->points.size(); i++)
        {
            mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
            mesh.getColors()[i] = ofColor(cloud->points[i].r,cloud->points[i].g,cloud->points[i].b);
        }
    }
    else
    {
        for(int y=0; y<(int)cloud->height; y++)
        {
            for(int x=0; x<(int)cloud->width; x++)
            {
                if(isnan(cloud->points[i].x) || isnan(cloud->points[i].y) || isnan(cloud->points[i].z))
                {
                    mesh.getVertices()[i] = ofVec3f(float(x)/float(cloud->width)*xfactor,float(y)/float(cloud->height)*yfactor,0);
                    mesh.getColors()[i] = ofColor(0,0,0);
                    n++;
                }
                else
                {
                    mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
                    mesh.getColors()[i] = ofColor(cloud->points[i].r,cloud->points[i].g,cloud->points[i].b);
                }
                i++;
                cout << ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor) << endl;
            }
        }
    }
}

void toOf(PointCloudPtr & cloud, ofMesh & mesh, float xfactor, float yfactor, float zfactor)
{
    mesh.clear();
    mesh.setMode(OF_PRIMITIVE_POINTS);
    mesh.getVertices().resize(cloud->points.size());
    if(cloud->is_dense)
    {
        for(int i=0; i<cloud->points.size(); i++)
        {
            mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
        }
    }
    else
    {
        int i=0, n=0;
        for(int y=0; y<(int)cloud->height; y++)
        {
            for(int x=0; x<(int)cloud->width; x++)
            {
                if(isnan(cloud->points[i].x) || isnan(cloud->points[i].y) || isnan(cloud->points[i].z))
                {
                    mesh.getVertices()[i] = ofVec3f(float(x)/float(cloud->width)*xfactor,float(y)/float(cloud->height)*yfactor,0);
                    n++;
                }
                else
                {
                    mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
                }
                i++;
            }
        }
    }
}

void toOf(ColorPointCloudPtr & cloud, SurfaceNormalsPtr & normals, ofMesh & mesh, float xfactor, float yfactor, float zfactor)
{
    mesh.setMode(OF_PRIMITIVE_POINTS);
    mesh.getVertices().resize(cloud->points.size());
    mesh.getColors().resize(cloud->points.size());
    if(cloud->is_dense)
    {
        for(int i=0; i<cloud->points.size(); i++)
        {
            mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
            mesh.getColors()[i] = ofColor(cloud->points[i].r,cloud->points[i].g,cloud->points[i].b);
        }
        if(normals)
        {
            mesh.getNormals().resize(normals->points.size());
            for(int i=0; i<normals->points.size(); i++)
            {
                mesh.getNormals()[i] = ofVec3f(normals->points[i].normal_x,normals->points[i].normal_y,normals->points[i].normal_z);
            }
        }
    }
    else
    {
        int i=0, n=0;
        for(int y=0; y<(int)cloud->height; y++)
        {
            for(int x=0; x<(int)cloud->width; x++)
            {
                if(isnan(cloud->points[i].x) || isnan(cloud->points[i].y) || isnan(cloud->points[i].z))
                {
                    mesh.getVertices()[i] = ofVec3f(float(x)/float(cloud->width)*xfactor,float(y)/float(cloud->height)*yfactor,0);
                    mesh.getColors()[i] = ofColor(0,0,0);
                    n++;
                }
                else
                {
                    mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
                    mesh.getColors()[i] = ofColor(cloud->points[i].r,cloud->points[i].g,cloud->points[i].b);
                }
                i++;
            }
        }
        for(int i=0; i<normals->points.size(); i++)
        {
            mesh.getNormals()[i] = ofVec3f(normals->points[i].normal_x,normals->points[i].normal_y,normals->points[i].normal_z);
        }
    }
}

void toOf(IntensityPointCloudPtr & cloud, ofMesh & mesh, float xfactor, float yfactor, float zfactor)
{
    mesh.setMode(OF_PRIMITIVE_POINTS);
    mesh.getVertices().resize(cloud->points.size());
    //mesh.getColors().resize(cloud->points.size());
    for(int i=0; i<cloud->points.size(); i++)
    {
        mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
        //mesh.getColors()[i] = ofFloatColor(cloud->points[i].intensity,cloud->points[i].intensity,cloud->points[i].intensity);
    }
}

void toOf(IntensityPointNormalsPtr & cloud, ofMesh & mesh, float xfactor, float yfactor, float zfactor)
{
    mesh.setMode(OF_PRIMITIVE_POINTS);
    mesh.getVertices().resize(cloud->points.size());
    mesh.getNormals().resize(cloud->points.size());
    //mesh.getColors().resize(cloud->points.size());
    for(int i=0; i<cloud->points.size(); i++)
    {
        mesh.getVertices()[i] = ofVec3f(cloud->points[i].x*xfactor,cloud->points[i].y*yfactor,cloud->points[i].z*zfactor);
        mesh.getNormals()[i] = ofVec3f(cloud->points[i].normal[0],cloud->points[i].normal[1],cloud->points[i].normal[2]);

        //mesh.getColors()[i] = ofFloatColor(cloud->points[i].intensity,cloud->points[i].intensity,cloud->points[i].intensity);
    }
}

void toPCL(const ofMesh & mesh, PointCloudPtr pc)
{
    pc->clear();
    for(int i=0; i<mesh.getVertices().size(); i++)
    {
        const ofVec3f & v = mesh.getVertices()[i];
        pc->push_back(PointT(v.x*0.001,v.y*0.001,v.z*0.001));
    }
}

void transform(PointCloudPtr cloud, ofMatrix4x4 matrix)
{
    assert(cloud);

    if (cloud->points.empty()) return;

    Eigen::Matrix4f mat;
    memcpy(&mat, matrix.getPtr(), sizeof(float) * 16);
    pcl::transformPointCloud(*cloud, *cloud, mat);
}

void addIndices(ofMesh & mesh, MeshPtr & triangles)
{
    mesh.getIndices().clear();
    mesh.setMode(OF_PRIMITIVE_TRIANGLES);
    for(int i=0; i<triangles->polygons.size(); i++)
    {
        for(int j=0; j<triangles->polygons[i].vertices.size(); j++)
        {
            mesh.addIndex(triangles->polygons[i].vertices[j]);
        }
    }
}

void smooth(ColorPointCloudPtr & cloud, ColorPointCloudPtr & mls_points, SurfaceNormalsPtr & mls_normals, float radius)
{
    // Create a KD-Tree
    pcl::search::KdTree<ColorPointT>::Ptr tree (new pcl::search::KdTree<ColorPointT>);
    tree->setInputCloud (cloud);

    // Init object (second point type is for the normals, even if unused)
    /* not working in 1.6
    		pcl::MovingLeastSquares<ColorPointT, NormalT> mls;


    		// Optionally, a pointer to a cloud can be provided, to be set by MLS
    		mls.setOutputNormals (mls_normals);

    		// Set parameters
    		mls.setInputCloud (cloud);
    		mls.setPolynomialFit (true);
    		mls.setSearchMethod (tree);
    		mls.setSearchRadius (radius);
    		//mls.setPolynomialOrder(10);
    		//mls.setSqrGaussParam(radius*radius);

    		// Reconstruct
    		mls.reconstruct (*mls_points);

    		// Concatenate fields for saving
    		//PointNormals mls_cloud;
    		//pcl::concatenateFields (mls_points, *mls_normals, mls_cloud);
    */
}

void surfelSmooth(ColorPointCloudPtr & cloud, ColorPointCloudPtr & mls_points, SurfaceNormalsPtr & mls_normals)
{
    //pcl::SurfelSmoothing smoother;
    //smoother.computeSmoothedCloud(mls_points,mls_normals)
}

void getMesh(PointCloudPtr & cloud, ofMesh & mesh)
{
    MeshPtr triangles(new Mesh);
    if(!cloud->is_dense)
    {
        pcl::OrganizedFastMesh<PointT> gp;

        gp.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
        gp.setInputCloud(cloud);
        gp.setTrianglePixelSize(3);
        gp.reconstruct(*triangles);
        toOf(cloud,mesh,1000,1000,1000);
        addIndices(mesh,triangles);
    }
    else
    {
        SurfaceNormalsPtr normals = calculateNormals(cloud);

        // Concatenate the XYZ and normal fields*
        IntensityPointNormalsPtr cloud_with_normals (new pcl::PointCloud<IntensityPointNormalT>);
        pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
        //* cloud_with_normals = cloud + normals

        // Create search tree*
        pcl::search::KdTree<IntensityPointNormalT>::Ptr tree2 (new pcl::search::KdTree<IntensityPointNormalT>);
        tree2->setInputCloud (cloud_with_normals);

        // Initialize objects
        pcl::GreedyProjectionTriangulation<IntensityPointNormalT> gp3;
        // Set the maximum distance between connected points (maximum edge length)
        gp3.setSearchRadius (1);

        // Set typical values for the parameters
        gp3.setMu (1.5);
        gp3.setMaximumNearestNeighbors (100);
        gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
        gp3.setMinimumAngle(M_PI/18); // 10 degrees
        gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
        gp3.setNormalConsistency(false);

        // Get result
        gp3.setInputCloud (cloud_with_normals);
        gp3.setSearchMethod (tree2);
        gp3.reconstruct (*triangles);

        toOf(cloud_with_normals,mesh,1000,1000,1000);
        addIndices(mesh,triangles);
    }
}


void marchingCubes(PointCloudPtr & cloud, ofMesh & mesh)
{
    SurfaceNormalsPtr normals = calculateNormals(cloud);
    MeshPtr triangles(new Mesh);

    // Concatenate the XYZ and normal fields*
    IntensityPointNormalsPtr cloud_with_normals (new IntensityPointNormals);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<IntensityPointNormalT>::Ptr tree2 (new pcl::search::KdTree<IntensityPointNormalT>);
    tree2->setInputCloud (cloud_with_normals);

    /* not working in 1.6
    		// Initialize objects
    		pcl::MarchingCubesGreedy<IntensityPointNormalT> mc;


    		// Get result
    		mc.setInputCloud (cloud_with_normals);
    		mc.setSearchMethod (tree2);
    		mc.reconstruct (*triangles);

    		toOf(cloud_with_normals,mesh,1000,1000,1000);
    		addIndices(mesh,triangles);
    */
}

SurfaceNormalsPtr calculateNormals(PointCloudPtr & cloud, float normal_radius_)
{
    pcl::search::KdTree<PointT>::Ptr search_method_xyz_;
    SurfaceNormalsPtr normals_;
    normals_ = SurfaceNormalsPtr (new SurfaceNormals);

    pcl::NormalEstimation<PointT, NormalT> norm_est;
    norm_est.setInputCloud (cloud);
    norm_est.setSearchMethod (search_method_xyz_);
    //norm_est.setRadiusSearch (normal_radius_);
    norm_est.setKSearch(20);
    norm_est.compute (*normals_);

    return normals_;
}


SurfaceNormalsPtr calculateNormalsOrdered(PointCloudPtr & cloud, pcl::IntegralImageNormalEstimation<PointT, NormalT>::NormalEstimationMethod normalEstimationMethod,float maxDepthChangeFactor, float normalSmoothingSize)
{
    SurfaceNormalsPtr normals (new SurfaceNormals ());

    pcl::IntegralImageNormalEstimation<PointT, NormalT> ne;
    ne.setNormalEstimationMethod (normalEstimationMethod);
    ne.setMaxDepthChangeFactor(maxDepthChangeFactor);
    ne.setNormalSmoothingSize(normalSmoothingSize);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    return normals;
}

SurfaceNormalsPtr calculateNormalsParallel(PointCloudPtr & cloud, float normal_radius_)
{
    pcl::search::KdTree<PointT>::Ptr search_method_xyz_;
    SurfaceNormalsPtr normals_;
    normals_ = SurfaceNormalsPtr (new SurfaceNormals);

    pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
    norm_est.setInputCloud (cloud);
    norm_est.setSearchMethod (search_method_xyz_);
//    norm_est.setRadiusSearch (normal_radius_);
    norm_est.setKSearch(20);
    norm_est.compute (*normals_);

    return normals_;
}

LocalDescriptorsPtr
computeLocalFeatures (const PointCloudPtr & cloudIn, const SurfaceNormalsPtr & normals,float feature_radius)
{
    LocalDescriptorsPtr features(new LocalDescriptors);
    pcl::search::KdTree<PointT>::Ptr search_method_xyz_;

    pcl::FPFHEstimation<PointT, NormalT, LocalDescriptorT> fpfh_est;
    fpfh_est.setInputCloud (cloudIn);
    fpfh_est.setInputNormals (normals);
    fpfh_est.setSearchMethod (search_method_xyz_);
    //fpfh_est.setRadiusSearch (feature_radius);
    fpfh_est.setKSearch (20);
    fpfh_est.compute (*features);
    return features;
}

LocalDescriptorsPtr
computeLocalFeatures (const PointNormalsPtr & cloudIn, float feature_radius)
{
    LocalDescriptorsPtr features(new LocalDescriptors);
    pcl::search::KdTree<PointNormalT>::Ptr search_method_xyz_;

    pcl::FPFHEstimation<PointNormalT, PointNormalT, LocalDescriptorT> fpfh_est;
    fpfh_est.setInputCloud (cloudIn);
    fpfh_est.setInputNormals (cloudIn);
    fpfh_est.setSearchMethod (search_method_xyz_);
    //fpfh_est.setRadiusSearch (feature_radius);
    fpfh_est.setKSearch (20);
    fpfh_est.compute (*features);
    return features;
}

void voxelGridFilter(const PointCloudPtr & cloudIn, PointCloud & cloudOut, float leafSize )
{
    pcl::VoxelGrid<PointT> p;
    p.setInputCloud (cloudIn);
    //p.setFilterLimits (0.0, 0.5);
    //p.setFilterFieldName ("z");
    p.setLeafSize (leafSize, leafSize, leafSize);
    p.filter(cloudOut);
}

void statisticalOutlierFilter(const PointCloudPtr & cloudIn, PointCloud & cloudOut )
{
    pcl::StatisticalOutlierRemoval<PointT> p;
    p.setInputCloud (cloudIn);
    p.setMeanK (50);
    p.setStddevMulThresh (1.0);
    p.filter(cloudOut);
}

void removeDistanteFilter(const PointCloudPtr & cloudIn, PointCloud & cloudOut , float depth_limit)
{
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud (cloudIn);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, depth_limit);
    pass.filter (cloudOut);
}

MeshPtr gridProjection(PointCloudPtr & cloud, float res, float padding)
{
    SurfaceNormalsPtr normals = calculateNormals(cloud,0.02);
    MeshPtr pm(new Mesh);
    IntensityPointNormalsPtr cloud_with_normals (new IntensityPointNormals);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    pcl::GridProjection<IntensityPointNormalT> gridProjection;
    pcl::search::KdTree<IntensityPointNormalT>::Ptr tree2 (new pcl::search::KdTree<IntensityPointNormalT>);
    tree2->setInputCloud (cloud_with_normals);
    // Set parameters
    gridProjection.setResolution(0.005);
    gridProjection.setPaddingSize(3);
    //gridProjection.setNearestNeighborNum(100);
    //gridProjection.setMaxBinarySearchLevel(10);

    gridProjection.setSearchMethod(tree2);
    gridProjection.setInputCloud(cloud_with_normals);
    gridProjection.reconstruct(*pm);
    return pm;
}

PointNormalsPtr movingLeastSquares(PointCloudPtr & cloud)
{
    PointCloud mls_points;
    SurfaceNormalsPtr mls_normals (new SurfaceNormals ());

    /* not working in 1.6

    		  pcl::MovingLeastSquares<PointT, NormalT> mls;
    		  boost::shared_ptr<vector<int> > indices (new vector<int>);


    		  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    		  tree->setInputCloud (cloud);

    		  // Set parameters
    		  mls.setInputCloud (cloud);
    		  //mls.setIndices (indices);
    		  mls.setPolynomialFit (true);
    		  mls.setSearchMethod (tree);
    		  mls.setSearchRadius (0.03);

    		  // Reconstruct
    		  mls.setOutputNormals (mls_normals);
    		  mls.reconstruct (mls_points);
    */
    PointNormalsPtr mls_cloud (new PointNormals ());
    pcl::concatenateFields (mls_points, *mls_normals, *mls_cloud);

    return mls_cloud;

}

MeshPtr marchingCubes(PointCloudPtr & cloud, float leafSize)
{
    SurfaceNormalsPtr normals = calculateNormals(cloud);
    MeshPtr pm (new Mesh);
    IntensityPointNormalsPtr cloud_with_normals (new IntensityPointNormals);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);

    /* not working in 1.6
    		pcl::MarchingCubesGreedy<IntensityPointNormalT> marchingCubes;
    		pcl::search::KdTree<IntensityPointNormalT>::Ptr tree2 (new pcl::search::KdTree<IntensityPointNormalT>);
    		tree2->setInputCloud (cloud_with_normals);

    		marchingCubes.setSearchMethod(tree2);
    		marchingCubes.setLeafSize(leafSize);
    		marchingCubes.setIsoLevel(0.5);   //ISO: must be between 0 and 1.0
    		marchingCubes.setInputCloud(cloud_with_normals);
    		marchingCubes.reconstruct(*pm);
    */
    return pm;
}

MeshPtr greedyProjectionTriangulation(PointCloudPtr & cloud)
{
    MeshPtr triangles(new Mesh);

    PointNormalsPtr cloud_with_normals = movingLeastSquares(cloud);

    // Create search tree*
    pcl::search::KdTree<PointNormalT>::Ptr tree2 (new pcl::search::KdTree<PointNormalT>);
    tree2->setInputCloud (cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<PointNormalT> gp3;
    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.3);

    // Set typical values for the parameters
    gp3.setMu (1.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (*triangles);

    return triangles;
}

PointCloudPtr
thresholdDepth (const PointCloudPtr & input, float min_depth, float max_depth)
{
    pcl::PassThrough<PointT> pass_through;
    pass_through.setInputCloud (input);
    pass_through.setFilterFieldName ("z");
    pass_through.setFilterLimits (min_depth, max_depth);
    PointCloudPtr thresholded (new PointCloud);
    pass_through.filter (*thresholded);

    return (thresholded);
}

/* Use a VoxelGrid filter to reduce the number of points */
PointCloudPtr
downsample (const PointCloudPtr & input, float leaf_size)
{
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud (input);
    voxel_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
    PointCloudPtr downsampled (new PointCloud);
    voxel_grid.filter (*downsampled);

    return (downsampled);
}

/* Use a RadiusOutlierRemoval filter to remove all points with too few local neighbors */
PointCloudPtr
removeOutliers (const PointCloudPtr & input, float radius, int min_neighbors)
{
    pcl::RadiusOutlierRemoval<PointT> radius_outlier_removal;
    radius_outlier_removal.setInputCloud (input);
    radius_outlier_removal.setRadiusSearch (radius);
    radius_outlier_removal.setMinNeighborsInRadius (min_neighbors);
    PointCloudPtr inliers (new PointCloud);
    radius_outlier_removal.filter (*inliers);

    return (inliers);
}


void pair_align(const PointCloudPtr cloud_src,
                const PointCloudPtr cloud_tgt,
                ofMatrix4x4 &final_transform,
                bool downsample,
                double voxel_leaf_size)
{
    //
    // Downsample for consistency and speed
    // \note enable this for large datasets
    PointCloudPtr src(new pcl::PointCloud<PointT>);
    PointCloudPtr tgt(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> grid;
    if (downsample) {
        grid.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        grid.setInputCloud(cloud_src);
        grid.filter(*src);
        std::cout << "Source cloud size before " << cloud_src->size() << " after " << src->size() << std::endl;
        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
        std::cout << "Target cloud size before " << cloud_tgt->size() << " after " << tgt->size() << std::endl;
    } else {
        src = cloud_src;
        tgt = cloud_tgt;
    }

    // Compute surface normals and curvature
    std::cout << "Compute Normals" << std::endl;
    PointNormalsPtr points_with_normals_src(
        new PointNormals);
    PointNormalsPtr points_with_normals_tgt(
        new PointNormals);

    SurfaceNormalsPtr normals_src = calculateNormalsParallel(src);
    SurfaceNormalsPtr normals_tgt = calculateNormalsParallel(tgt);

    pcl::concatenateFields (*src, *normals_src, *points_with_normals_src);
    pcl::concatenateFields (*tgt, *normals_tgt, *points_with_normals_tgt);

    // ### Feature Estimation ########################################
    std::cout << "Estimate Features" << std::endl;

    // Output datasets
    LocalDescriptorsPtr source_features(
        new LocalDescriptors);
    LocalDescriptorsPtr target_features(
        new LocalDescriptors);

    source_features = computeLocalFeatures(points_with_normals_src);
    target_features = computeLocalFeatures(points_with_normals_tgt);


    // ############################################################################

    final_transform = sampleConsensusInitialAlignment(points_with_normals_src, points_with_normals_tgt, source_features, target_features);

}




ofMatrix4x4 sampleConsensusInitialAlignment(PointNormalsPtr points_with_normals_src,
        PointNormalsPtr points_with_normals_tgt,
        LocalDescriptorsPtr source_features,
        LocalDescriptorsPtr target_features,
        double ia_min_sample_distance,
        float ia_max_distance,
        int ia_iterations)
{
    pcl::SampleConsensusInitialAlignment<PointNormalT, PointNormalT, LocalDescriptorT> sac;
    sac.setMinSampleDistance(ia_min_sample_distance);
    sac.setMaxCorrespondenceDistance(ia_max_distance);
    sac.setMaximumIterations(ia_iterations);
    sac.setInputSource(points_with_normals_src);
    sac.setSourceFeatures(source_features);
    sac.setInputTarget(points_with_normals_tgt);
    sac.setTargetFeatures(target_features);

    PointNormalsPtr pre_aligned_source(new PointNormals);
    sac.align(*pre_aligned_source);

    Eigen::Matrix4f initial_T = sac.getFinalTransformation();

    ofMatrix4x4 final_transform;
    memcpy(final_transform.getPtr(), &initial_T, sizeof(float) * 16);
    return final_transform;
}

}
