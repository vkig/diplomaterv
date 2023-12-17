#include "livox_lidar_time_fixer_node.hpp"

// see https://github.com/Livox-SDK/livox_ros_driver/issues/43

LLTF::LLTF():nh("~")
{
    topic_sub_LidarImu = "/livox/imu/old";
    topic_pub_LidarImu = "/livox/imu";
    topic_sub_LidarCloud = "/livox/lidar/old";
    topic_pub_LidarCloud = "/livox/lidar";
    
    nh.getParam("Topic_Sub_LidarImu", topic_sub_LidarImu);
    nh.getParam("Topic_Pub_LidarImu", topic_pub_LidarImu);
    nh.getParam("Topic_Sub_LidarCloud", topic_sub_LidarCloud);
    nh.getParam("Topic_Pub_LidarCloud", topic_pub_LidarCloud);

    int queue_sub = 1;
    int queue_pub = 1;
    sub_LidarImu = nh.subscribe<sensor_msgs::Imu>(topic_sub_LidarImu, queue_sub, &LLTF::Callback_LidarImu, this);
    pub_LidarImu = nh.advertise<sensor_msgs::Imu>(topic_pub_LidarImu, queue_pub);
    sub_LidarCloud = nh.subscribe<sensor_msgs::PointCloud2>(topic_sub_LidarCloud, queue_sub, &LLTF::Callback_LidarCloud, this);
    pub_LidarCloud = nh.advertise<sensor_msgs::PointCloud2>(topic_pub_LidarCloud, queue_pub);
}

void LLTF::Callback_LidarImu(const sensor_msgs::ImuConstPtr& msg)
{
    if (pub_LidarImu.getNumSubscribers() == 0) return;
    sensor_msgs::Imu msg_new;
    msg_new = *msg;
    msg_new.header.stamp = ros::Time::now();
    pub_LidarImu.publish(msg_new);
}

void LLTF::Callback_LidarCloud(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    if (pub_LidarCloud.getNumSubscribers() == 0) return;
    sensor_msgs::PointCloud2 msg_new;
    msg_new = *msg;
    msg_new.header.stamp = ros::Time::now();
    pub_LidarCloud.publish(msg_new);
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"livox_lidar_timefixer_node");

    LLTF lltf;

    ros::spin();

    return 0;
}
