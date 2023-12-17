#include "data_converter_node.hpp"

DataConverter::DataConverter() : nodeHandle("~"), rate(30.0)
{
    ros::Time lastTimestamp = ros::Time(0);

    frame = nodeHandle.param<std::string>("/Frame", "livox_frame");
    frequency = nodeHandle.param("/Frequency", 30.0);

    subTopicLidarCloud = nodeHandle.param<std::string>("/Topic_Pub_FilteredCloud", "/filtered_cloud");
    pubTopicProcessedData = "/processed_data";

    minPointOfBBX = nodeHandle.param("/BackgroundModel_Filter_BBX_min", std::vector<double>({10.0, -20.0, -7.0}));
    maxPointOfBBX = nodeHandle.param("/BackgroundModel_Filter_BBX_max", std::vector<double>({34.0, 2.5, 3.0}));

    int subQueueSize = 1;
    int pubQueueSize = 1;

    subLidarCloud = nodeHandle.subscribe(subTopicLidarCloud, subQueueSize, &DataConverter::callbackLidarCloud, this);
    pubProcessedData = nodeHandle.advertise<bird_vs_drone_msgs::PointCloudData>(pubTopicProcessedData, pubQueueSize);

    networkInputSize = nodeHandle.param("Network_Input_Size", 128);
}

DataConverter::~DataConverter(){}

void DataConverter::callbackLidarCloud(const sensor_msgs::PointCloud2 &msg)
{
    lastTimestamp = ros::Time::now();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::fromROSMsg(msg, *cloud);

    if(cloud->size() > networkInputSize)
    {
        downSampling(cloud);   
    }

    std::vector<bird_vs_drone_msgs::PointXYZI_msg> points;
    for (int i = 0; i < cloud->points.size(); i++)
    {
        bird_vs_drone_msgs::PointXYZI_msg point;
        point.x = cloud->points[i].x;
        point.y = cloud->points[i].y;
        point.z = cloud->points[i].z;
        point.i = cloud->points[i].intensity;
        points.push_back(point);
    }

    data = points;
    publishProcessedData(msg.header.stamp);
}

void DataConverter::loop()
{
    while (nodeHandle.ok())
    {
        ros::spinOnce();

        rate.sleep();
    }
}

void DataConverter::downSampling(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
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

void DataConverter::publishProcessedData(ros::Time timestamp)
{
    bird_vs_drone_msgs::PointCloudData toPublish;
    toPublish.header.stamp = timestamp;
    toPublish.header.frame_id = frame;
    toPublish.points = data;
    pubProcessedData.publish(toPublish);
    std::cout << "DataConverter process time: " << (ros::Time::now() - lastTimestamp).toSec() << std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"data_converter_node");

    DataConverter dataConverter;

    ROS_INFO("Data converter node is waiting for pointcloud...");
    dataConverter.loop();

    return 0;
}