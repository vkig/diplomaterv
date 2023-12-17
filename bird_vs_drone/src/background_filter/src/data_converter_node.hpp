#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <jsk_rviz_plugins/OverlayText.h>
#include <tf/transform_listener.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/filters/crop_box.h>
#include <bird_vs_drone_msgs/PointCloudData.h>
#include <pcl/filters/radius_outlier_removal.h>

class DataConverter {
    public:
        DataConverter();
        ~DataConverter();
        void callbackLidarCloud(const sensor_msgs::PointCloud2 &msg);
        void loop();
        void downSampling(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
        void publishProcessedData(ros::Time timestamp);
    private:
        ros::NodeHandle nodeHandle;
        ros::Rate rate;
        ros::Time lastTimestamp;
        std::string frame;
        double frequency;
        std::vector<double> minPointOfBBX;
        std::vector<double> maxPointOfBBX;
        std::string subTopicLidarCloud;
        std::string pubTopicProcessedData;
        ros::Subscriber subLidarCloud;
        ros::Publisher pubProcessedData;
        std::vector<bird_vs_drone_msgs::PointXYZI_msg> data;
        int networkInputSize;
        struct DistanceNode {
            int index;
            float distance;
            
            bool operator<(const DistanceNode& rhs) const
            {
                if (distance == rhs.distance)
                {
                    return index < rhs.index;
                }
                return distance < rhs.distance;
            }
        };
};