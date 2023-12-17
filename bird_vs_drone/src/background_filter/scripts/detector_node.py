#!/usr/bin/env python
import torch
from model import *
from data_loader import *
import rospy
from bird_vs_drone_msgs.msg import PointCloudData
from visualization_msgs.msg import Marker, MarkerArray
from utils import *

class DetectorNode:
    def __init__(self):
        rospy.init_node("detector_node")
        self.rate = rospy.Rate(30)
        self.net = BirdNet()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net.load_state_dict(torch.load(os.path.join(SAVE_PATH, str(67) + ".pt"), map_location=self.device))
        self.net.eval()
        self.net.to(self.device)
        self.pub = rospy.Publisher(BOUNDING_BOXES_TOPIC, MarkerArray, queue_size=10)
        self.sub = rospy.Subscriber(PROCESSED_DATA_TOPIC, PointCloudData, self.callback)
        self.inference_times = []
        self.index = 0
        self.prev_marker_count = 0
        
    def callback(self, PointCloudData):
        tick = rospy.Time.now()
        points = PointCloudData.points
        input, bbxs = find_object_points_and_bbx(points)
        if input == None:
            bbx_msg = MarkerArray()
            bbx_msg.markers = []
            bbx_marker_count_tmp = len(bbx_msg.markers)
            for i in range(len(bbx_msg.markers), self.prev_marker_count):
                marker = Marker()
                marker.id = i
                marker.header.frame_id = FRAME_NAME
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.CUBE
                marker.action = Marker.DELETE
                bbx_msg.markers.append(marker)
            self.prev_marker_count = bbx_marker_count_tmp
            self.pub.publish(bbx_msg)
            return
        input = input.to(self.device)
        pred = self.net(input)
        pred_choice = pred.data.max(1)[1]
        bbx_msg = MarkerArray()
        bbx_msg.markers = []
        for i in range(0, len(pred_choice)):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = FRAME_NAME
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = bbxs[i][0][0] 
            marker.pose.position.y = bbxs[i][0][1]
            marker.pose.position.z = bbxs[i][0][2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = bbxs[i][1][0]
            marker.scale.y = bbxs[i][1][1]
            marker.scale.z = bbxs[i][1][2]
            marker.color.a = 1.0
            #marker.lifetime = rospy.Duration(0.1)
            if pred_choice[i] == 0:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif pred_choice[i] == 1:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            bbx_msg.markers.append(marker)
            self.inference_times.append(rospy.Time.now() - PointCloudData.header.stamp)
            if len(self.inference_times) == 10:
                self.inference_times.pop(0)
        bbx_marker_count_tmp = len(bbx_msg.markers)
        for i in range(len(bbx_msg.markers), self.prev_marker_count):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = FRAME_NAME
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.CUBE
            marker.action = Marker.DELETE
            bbx_msg.markers.append(marker)
        self.prev_marker_count = bbx_marker_count_tmp
        self.pub.publish(bbx_msg)
        tock = rospy.Time.now()
        avg_inference_time = rospy.Duration(0, 0)
        if len(self.inference_times) != 0:
            avg_inference_time_s = int(sum([t.secs for t in self.inference_times]) / len(self.inference_times))
            avg_inference_time_ns = int(sum([t.nsecs for t in self.inference_times]) / len(self.inference_times))
            avg_inference_time = rospy.Duration(avg_inference_time_s, avg_inference_time_ns)
        self.index += 1
        if self.index % 2 == 0:
            print("Average inference time: ", avg_inference_time.to_sec(), "s")
            print("Actual inference time: ", (tock - tick).to_sec(), "s")
            print("Whole process time: ", self.inference_times[-1].to_sec(), "s")
            self.index = 0
        self.rate.sleep()
        
        
if __name__ == "__main__":
    node = DetectorNode()
    print("Detector node started...")
    while not rospy.is_shutdown():
        node.rate.sleep()