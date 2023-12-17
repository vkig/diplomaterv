#include "background_filter_node.hpp"

BackgroundFilter::BackgroundFilter(): nodeHandle("~"), rate(30.0), bgFilteredCloud(new pcl::PointCloud<pcl::PointXYZI>)
{
    ros::Time lastTimestamp = ros::Time(0);
    bgModelBuildStart = ros::Time(0);
    bgModelBuilding = true;

    std::vector<std::string> paramNames;
    bool result = nodeHandle.getParamNames(paramNames);

    bgModelResolution = nodeHandle.param("/BackgroundModel_Resolution", 0.1);
    bgModelBuildingTime = nodeHandle.param("/BackgroundModel_BuildingTime", 15.0);
    double thresholdTemp = nodeHandle.param("/Threshold_Seconds_For_Background_Filter", 10.0);
    bgModelThresholdSeconds = ros::Duration(thresholdTemp);
    frame = nodeHandle.param<std::string>("/Frame", "livox_frame");
    frequency = nodeHandle.param("/Frequency", 30.0);

    bgModelInflate = nodeHandle.param("/BackgroundModel_Inflate", true);
    bgModelBBX = nodeHandle.param("/BackgroundModel_Filter_BBX", true);

    minPointOfBBX = nodeHandle.param("/BackgroundModel_Filter_BBX_min", std::vector<double>({10.0, -20.0, -7.0}));
    maxPointOfBBX = nodeHandle.param("/BackgroundModel_Filter_BBX_max", std::vector<double>({34.0, 2.5, 3.0}));
    networkInputSize = nodeHandle.param("Network_Input_Size", 128);

    subTopicLidarCloud = nodeHandle.param<std::string>("/Topic_Sub_LidarCloud", "/livox/lidar");
    pubTopicFilteredCloud = nodeHandle.param<std::string>("/Topic_Pub_FilteredCloud", "/filtered_cloud");
    pubBgModelTopic = nodeHandle.param<std::string>("/Topic_Pub_BackgroundModel", "/background_model");
    pubBoundingBoxTopic = nodeHandle.param<std::string>("/Topic_Pub_BoundingBox", "/bounding_box");
    pubTopicProcessedData = "/processed_data";

    int subQueueSize = 1;
    int pubQueueSize = 1;

    subLidarCloud = nodeHandle.subscribe(subTopicLidarCloud, subQueueSize, &BackgroundFilter::callbackLidarCloud, this);
    pubFilteredCloud = nodeHandle.advertise<sensor_msgs::PointCloud2>(pubTopicFilteredCloud, pubQueueSize);
    pubBgModel = nodeHandle.advertise<sensor_msgs::PointCloud2>(pubBgModelTopic, pubQueueSize, true);
    pubBoundingBox = nodeHandle.advertise<visualization_msgs::Marker>(pubBoundingBoxTopic, pubQueueSize, true);
    pubProcessedData = nodeHandle.advertise<bird_vs_drone_msgs::PointCloudData>(pubTopicProcessedData, pubQueueSize);

    visualization_msgs::Marker boundingBox = visualization_msgs::Marker();
    boundingBox.type = visualization_msgs::Marker::LINE_LIST;
    boundingBox.id = 0;
    boundingBox.scale.x = 0.1;
    boundingBox.color.r = 1.0;
    boundingBox.color.a = 1.0;
    boundingBox.header.frame_id = frame;
    boundingBox.header.stamp = ros::Time::now();
    std::vector<std::vector<double>> bbxPoints = {minPointOfBBX, maxPointOfBBX};
    for (std::vector<size_t> index : std::vector<std::vector<size_t>>{{0, 0, 0}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1}})
    {
        geometry_msgs::Point base;
        base.x = bbxPoints[index[0]][0];
        base.y = bbxPoints[index[1]][1];
        base.z = bbxPoints[index[2]][2];
        for (size_t i = 0; i < 3; i++)
        {
            geometry_msgs::Point other;
            other.x = bbxPoints[(index[0] + 1 - (i != 0)) % 2][0];
            other.y = bbxPoints[(index[1] + 1 - (i != 1)) % 2][1];
            other.z = bbxPoints[(index[2] + 1 - (i != 2)) % 2][2];
            boundingBox.points.push_back(base);
            boundingBox.points.push_back(other);       
        }
    }
    pubBoundingBox.publish(boundingBox);

    voxels = std::vector<std::vector<std::vector<VoxelNode>>>{static_cast<size_t>((maxPointOfBBX[0] - minPointOfBBX[0]) / bgModelResolution),
                        std::vector<std::vector<VoxelNode>>{static_cast<size_t>((maxPointOfBBX[1] - minPointOfBBX[1]) / bgModelResolution),
                        std::vector<VoxelNode>{static_cast<size_t>((maxPointOfBBX[2] - minPointOfBBX[2]) / bgModelResolution), VoxelNode{}}}};
    std::cout << "x size: " << voxels.size() << "; y size: " << voxels[0].size() << "; z size: " << voxels[0][0].size() << std::endl;
    maxNumberOfPoints = 0;
}

BackgroundFilter::~BackgroundFilter(){}

void BackgroundFilter::callbackLidarCloud(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    auto start = std::chrono::steady_clock::now();
    cloudStamp = ros::Time::now();
    if (lastTimestamp == ros::Time(0)) lastTimestamp = ros::Time::now();
    if (bgModelBuildStart == ros::Time(0)){
        bgModelBuildStart = ros::Time::now();
    }
    // convert PointCloud
    pcl::PCLPointCloud2 pclPointCloud2;
    pcl_conversions::toPCL(*msg, pclPointCloud2);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(pclPointCloud2, *cloud);

    // filter PointCloud based on the area of interest
    pcl::CropBox<pcl::PointXYZI> cropBox{};
    cropBox.setMin(Eigen::Vector4f{float(minPointOfBBX[0] + bgModelResolution), float(minPointOfBBX[1] + bgModelResolution), float(minPointOfBBX[2] + bgModelResolution), 0.0f});
    cropBox.setMax(Eigen::Vector4f{float(maxPointOfBBX[0] - bgModelResolution), float(maxPointOfBBX[1] - bgModelResolution), float(maxPointOfBBX[2] - bgModelResolution), 300.0f});
    cropBox.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr croppedCloud(new pcl::PointCloud<pcl::PointXYZI>);
    cropBox.filter(*croppedCloud);

    bgFilteredCloud->clear();
    ros::Time now = ros::Time::now();
    std::vector<bird_vs_drone_msgs::PointXYZI_msg> points;
    // iterate through point cloud
    for (auto &&point : *croppedCloud)
    {
        std::array<size_t, 3> index = pointToIndex(point);
        if(bgModelBuilding)
        {
            if(voxels[index[0]][index[1]][index[2]].lastMsg != ros::Time(0))
            {
                // find the maximum time duration between two data in the same voxel
                if(now - voxels[index[0]][index[1]][index[2]].lastMsg > voxels[index[0]][index[1]][index[2]].maxInterval)
                {
                    voxels[index[0]][index[1]][index[2]].maxInterval = now - voxels[index[0]][index[1]][index[2]].lastMsg;
                }
            } else {
                voxels[index[0]][index[1]][index[2]].maxInterval = now - bgModelBuildStart;
            }
            voxels[index[0]][index[1]][index[2]].lastMsg = now;
        } else {
            if(!voxels[index[0]][index[1]][index[2]].background)
            {
                bgFilteredCloud->push_back(point);
                bird_vs_drone_msgs::PointXYZI_msg point_xyzi;
                point_xyzi.x = point.x;
                point_xyzi.y = point.y;
                point_xyzi.z = point.z;
                point_xyzi.i = point.intensity;
                points.push_back(point_xyzi);
            }
        }
    }

    if(bgModelBuilding){
        bgFilteredCloud = croppedCloud;
    } else {
        if(maxNumberOfPoints < bgFilteredCloud->size()){
            maxNumberOfPoints = bgFilteredCloud->size();
            std::cout << "Max number of points changed: " << maxNumberOfPoints << std::endl;
        }
    }
    
    data = points;

    publishProcessedData(cloudStamp);
    publishFilteredPointCloud(cloudStamp);

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

void BackgroundFilter::loop()
{
    // Main loop
    while (nodeHandle.ok())
    {
        // Check for new messages
        ros::spinOnce();

        // execute one loop
        execute();

        // sleep to match pre-defined frequency
        rate.sleep();
    }
}

void BackgroundFilter::execute()
{
    if(bgModelBuilding) {
        // if no point cloud message arrived yet return
        if (lastTimestamp == ros::Time(0)) return;
        // check passed time since beginning of model building
        ros::Duration time_passed = ros::Time::now() - lastTimestamp;
        ROS_INFO_STREAM("Building background model... " << time_passed.toSec() << "/" << bgModelBuildingTime << "s");
        // Build background model until threshold time over
        if (time_passed.toSec() > bgModelBuildingTime)
        {
            bgModelBuilding = false;
            lastTimestamp = ros::Time::now();
            ROS_INFO_STREAM("Building background model done.");
            publishBgModel();
        }
        return;
    } else {
        // update last timestamp
        lastTimestamp = ros::Time::now();
    }
}

void BackgroundFilter::publishBgModel()
{
    auto start = std::chrono::steady_clock::now();
    size_t count = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr backgroundModelCloud{new pcl::PointCloud<pcl::PointXYZ>};
    std::vector<std::array<size_t, 3>> indices{};
    ros::Time now = ros::Time::now();
    for (size_t x = 0; x < voxels.size(); x++)
    {
        for (size_t y = 0; y < voxels[x].size(); y++)
        {
            for (size_t z = 0; z < voxels[x][y].size(); z++)
            {
                if(now - voxels[x][y][z].lastMsg > voxels[x][y][z].maxInterval)
                {
                    voxels[x][y][z].maxInterval = now - voxels[x][y][z].lastMsg;
                }
                if(voxels[x][y][z].maxInterval < ros::Duration(bgModelThresholdSeconds)){
                    voxels[x][y][z].background = true;
                }
                if(voxels[x][y][z].background){
                    if(bgModelInflate){
                        // inflate bgModel
                        for (auto &&dx : {-1, 0, 1}) for (auto &&dy : {-1, 0, 1}) for (auto &&dz : {-1, 0, 1})
                        {
                            if(dx == dy && dy == dz && dz == 0) continue;
                            if(x + dx < 0 || x + dx >= voxels.size() || y + dy < 0 || y + dy >= voxels[x].size() || z + dz < 0 || z + dz >= voxels[x][y].size()) continue;
                            if(!voxels[x+dx][y+dy][z+dz].background){
                                indices.push_back({x+dx, y+dy, z+dz});
                                pcl::PointXYZ position = indexToCenterPoint(std::array<size_t, 3>{x+dx, y+dy, z+dz});
                                backgroundModelCloud->push_back(position);
                            }
                        }
                    }
                    pcl::PointXYZ position = indexToCenterPoint(std::array<size_t, 3>{x, y, z});
                    backgroundModelCloud->push_back(position);
                    count++;
                }
            }
        }
    }
    for (auto &&index : indices)
    {
        voxels[index[0]][index[1]][index[2]].background = true;
    }
    count += indices.size();
    
    auto end = std::chrono::steady_clock::now();
    std::cout << "BG model inflation, publication time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    std::cout << count << " <- The number of markers" << std::endl;
    std::cout << backgroundModelCloud->size() << " <- the size of the background model cloud" << std::endl;
    sensor_msgs::PointCloud2 message;
    pcl::toROSMsg(*backgroundModelCloud, message);
    message.header.stamp = ros::Time::now();
    message.header.frame_id = frame;
    pubBgModel.publish(message);
}

void BackgroundFilter::downSampling(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> filter;
    filter.setInputCloud(cloud);
    filter.setRadiusSearch(0.5);
    filter.setMinNeighborsInRadius(2);
    filter.filter(*cloud);
    if(cloud->size() > networkInputSize)
    {
        std::set<DistanceNode> distances;
        pcl::PointXYZI origin;
        origin.x = minPointOfBBX[0] + maxPointOfBBX[0] / 2;
        origin.y = minPointOfBBX[1] + maxPointOfBBX[1] / 2;
        origin.z = minPointOfBBX[2] + maxPointOfBBX[2] / 2;
        for(int i = 0; i < cloud->size(); i++){
            float distance = sqrt(pow(cloud->points[i].x - origin.x, 2) + pow(cloud->points[i].y - origin.y, 2) + pow(cloud->points[i].z - origin.z, 2));
            distances.insert(DistanceNode{i, distance});
        }
        std::set<int> indices_to_remove;
        for(auto node : distances){
            indices_to_remove.insert(node.index);
            if(indices_to_remove.size() == (cloud->size() - networkInputSize))
                break;
        }
        for(auto actual = (--indices_to_remove.end()); actual != indices_to_remove.begin(); actual--){
            cloud->erase(cloud->begin() + *actual);
        }
        cloud->erase(cloud->begin() + *indices_to_remove.begin());
    }
    assert(cloud->size() <= networkInputSize);
}

void BackgroundFilter::publishProcessedData(ros::Time timestamp)
{
    auto start = std::chrono::steady_clock::now();
    bird_vs_drone_msgs::PointCloudData toPublish;
    toPublish.header.stamp = timestamp;
    toPublish.header.frame_id = frame;
    toPublish.points = data;
    pubProcessedData.publish(toPublish);
    std::cout << "BackgroundFilter conversion process time: " << (ros::Time::now() - cloudStamp).toSec() << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::cout << "ProcessedData publication time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

void BackgroundFilter::publishFilteredPointCloud(ros::Time timestamp)
{
    auto start = std::chrono::steady_clock::now();
    sensor_msgs::PointCloud2 message;
    pcl::toROSMsg(*bgFilteredCloud, message);
    message.header.stamp = timestamp;
    message.header.frame_id = frame;
    pubFilteredCloud.publish(message);
    std::cout << "BackgroundFilter process time: " << (ros::Time::now() - cloudStamp).toSec() << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::cout << "BackgroundFilter publication time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

std::array<size_t, 3> BackgroundFilter::pointToIndex(pcl::PointXYZI &point)
{ 
    return std::array<size_t, 3>{static_cast<size_t>((point.x - minPointOfBBX[0]) / bgModelResolution),
                                static_cast<size_t>((point.y - minPointOfBBX[1]) / bgModelResolution),
                                static_cast<size_t>((point.z - minPointOfBBX[2]) / bgModelResolution)};
}

pcl::PointXYZ BackgroundFilter::indexToCenterPoint(std::array<size_t, 3> index)
{
    std::array<double, 3> indexDouble;
    for (size_t i = 0; i < index.size(); i++)
    {
        indexDouble[i] = static_cast<double>(index[i]) * bgModelResolution; 
    }
    indexDouble[0] += minPointOfBBX[0] + bgModelResolution / 2;
    indexDouble[1] += minPointOfBBX[1] + bgModelResolution / 2;
    indexDouble[2] += minPointOfBBX[2] + bgModelResolution / 2;
    return pcl::PointXYZ(static_cast<float>(indexDouble[0]), static_cast<float>(indexDouble[1]), static_cast<float>(indexDouble[2]));
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"background_filter_node");

    BackgroundFilter backgroundFilter;

    ROS_INFO("Background filter node is waiting for pointcloud...");
    backgroundFilter.loop();

    return 0;
}
