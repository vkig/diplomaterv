#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

class LLTF
{
    private:
        ros::NodeHandle nh;
        ros::Subscriber sub_LidarImu;
        ros::Publisher pub_LidarImu;
        ros::Subscriber sub_LidarCloud;
        ros::Publisher pub_LidarCloud;
        std::string topic_sub_LidarImu;
        std::string topic_pub_LidarImu;
        std::string topic_sub_LidarCloud;
        std::string topic_pub_LidarCloud;
    public:
        LLTF();
        void Callback_LidarImu(const sensor_msgs::ImuConstPtr& msg);
        void Callback_LidarCloud(const sensor_msgs::PointCloud2ConstPtr& msg);
};