#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_srvs/Trigger.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_listener.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/filters/crop_box.h>
#include <visualization_msgs/MarkerArray.h>
#include <bird_vs_drone_msgs/PointXYZI_msg.h>
#include <bird_vs_drone_msgs/PointCloudData.h>
#include <pcl/filters/radius_outlier_removal.h>

class BackgroundFilter {
    public:
        BackgroundFilter();
        ~BackgroundFilter();
        void callbackLidarCloud(const sensor_msgs::PointCloud2ConstPtr &msg);
        void loop();
        void execute();
        void publishBgModel();
        void publishFilteredPointCloud(ros::Time timestamp);
        std::array<size_t, 3> pointToIndex(pcl::PointXYZI& point);
        pcl::PointXYZ indexToCenterPoint(std::array<size_t, 3> index);
        void downSampling(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
        void publishProcessedData(ros::Time timestamp);
    private:
        ros::NodeHandle nodeHandle;
        ros::Rate rate;
        ros::Time lastTimestamp;
        ros::Time cloudStamp;
        ros::Time bgModelBuildStart;
        bool bgModelBuilding;
        std::string frame;
        double frequency;
        double bgModelResolution;
        double bgModelBuildingTime;
        ros::Duration bgModelThresholdSeconds;
        bool bgModelInflate;
        bool bgModelBBX;
        std::vector<double> minPointOfBBX;
        std::vector<double> maxPointOfBBX;
        std::string subTopicLidarCloud;
        std::string pubTopicFilteredCloud;
        std::string pubBgModelTopic;
        std::string pubBoundingBoxTopic;
        std::string pubTopicProcessedData;
        ros::Subscriber subLidarCloud;
        ros::Publisher pubFilteredCloud;
        ros::Publisher pubBgModel;
        ros::Publisher pubBoundingBox;
        ros::Publisher pubProcessedData;
        pcl::PointCloud<pcl::PointXYZI>::Ptr bgFilteredCloud;
        struct VoxelNode {
            ros::Duration maxInterval = ros::Duration(0);
            ros::Time lastMsg = ros::Time(0);
            bool background = false;
        };
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
        int networkInputSize;
        std::vector<bird_vs_drone_msgs::PointXYZI_msg> data;
        std::vector<std::vector<std::vector<VoxelNode>>> voxels;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        size_t maxNumberOfPoints;
};