import random
import struct
from typing import List
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from config import *


class Point:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0, intensity = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity


class Box:
    def __init__(self):
        self.x_min = 0.0
        self.x_max = 0.0
        self.y_min = 0.0
        self.y_max = 0.0
        self.z_min = 0.0
        self.z_max = 0.0


def filter_points_in_box(points: List[Point], box: Box):
    points_in_the_box = []
    for point in points:
        if box.x_min <= point.x <= box.x_max and \
                box.y_min <= point.y <= box.y_max and \
                box.z_min <= point.z <= box.z_max:
            points_in_the_box.append(point)
    return points_in_the_box


def time_from_annotation_ns(annotation, track_idx, label_idx):
    return annotation["tracks"][track_idx]["track"][label_idx]["header"]["stamp"]["secs"], \
        annotation["tracks"][track_idx]["track"][label_idx]["header"]["stamp"]["nsecs"]


def time_from_msg_ns(msg):
    return msg.header.stamp.secs * 1e9 + msg.header.stamp.nsecs


def parse_annotation(annotation):
    result = Box()
    result.x_min = annotation["translation"]["x"] - annotation["box"]["length"] / 2
    result.x_max = annotation["translation"]["x"] + annotation["box"]["length"] / 2
    result.y_min = annotation["translation"]["y"] - annotation["box"]["width"] / 2
    result.y_max = annotation["translation"]["y"] + annotation["box"]["width"] / 2
    result.z_min = annotation["translation"]["z"] - annotation["box"]["height"] / 2
    result.z_max = annotation["translation"]["z"] + annotation["box"]["height"] / 2
    return result


def parse_point(point):
    result = Point()
    result.x = struct.unpack_from('f', point, 0)[0]
    result.y = struct.unpack_from('f', point, 4)[0]
    result.z = struct.unpack_from('f', point, 8)[0]
    result.intensity = struct.unpack_from('f', point, 16)[0]
    return result


def parse_box_from_database(annotation):
    result = Box()
    result.x_min = annotation["x"] - annotation["length"] / 2
    result.x_max = annotation["x"] + annotation["length"] / 2
    result.y_min = annotation["y"] - annotation["width"] / 2
    result.y_max = annotation["y"] + annotation["width"] / 2
    result.z_min = annotation["z"] - annotation["height"] / 2
    result.z_max = annotation["z"] + annotation["height"] / 2
    return result


def parse_points_from_database(points):
    result = []
    for i in range(len(points["x"])):
        point = Point()
        point.x = points["x"][i]
        point.y = points["y"][i]
        point.z = points["z"][i]
        point.intensity = points["intensity"][i]
        result.append(point)
    return result


def plot_points_with_annotations(points, annotations, filename, distance, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 120)
    ax.set_ylim(-20, 30)
    ax.set_zlim(-6, 15)
    ax.set_title(filename + ", " + str(distance) + " m" + ", " + label)
    for i in range(len(points)):
        ax.scatter(points[i].x, points[i].y, points[i].z, c='b', marker='o')
    for i in range(len(annotations)):
        box = parse_box_from_database(annotations.iloc[i])
        ax.scatter(box.x_min, box.y_min, box.z_min, c='r', marker='o')
        ax.scatter(box.x_max, box.y_max, box.z_max, c='r', marker='o')
    plt.show()


def find_object_points_and_bbx(points):
    point_xyzi = np.array([[points[i].x, points[i].y, points[i].z, points[i].i] for i in range(len(points))])
    if len(point_xyzi) == 0:
        return None, None
    result = DBSCAN(eps=CLUSTERING_EPS, min_samples=POINTS_IN_THE_BOX_THRESHOLD).fit(point_xyzi[:, :3])
    num_of_clusters = len(set(result.labels_))
    bounding_boxes = []
    if -1 in result.labels_:
        num_of_clusters -= 1
    data = [[] for i in range(num_of_clusters)]
    for i in range(result.labels_.shape[0]):
        if result.labels_[i] != -1:
            data[result.labels_[i]].append(point_xyzi[i])
    input_data = []
    for i in range(len(data)):
        if len(data[i]) > INPUT_SHAPE[1]:
            new_data = []
            data_indices = [i for i in range(len(data[i]))]
            random.shuffle(data_indices)
            for j in range(INPUT_SHAPE[1]):
                new_data.append(data[i][data_indices[j]])
            data[i] = new_data
            
        point_data = torch.zeros(INPUT_SHAPE)
        x_min = X_MAX
        x_max = X_MIN
        y_min = Y_MAX
        y_max = Y_MIN
        z_min = Z_MAX
        z_max = Z_MIN
        for j in range(len(data[i])):
            if data[i][j][0] < x_min:
                x_min = data[i][j][0]
            if data[i][j][0] > x_max:
                x_max = data[i][j][0]
            if data[i][j][1] < y_min:
                y_min = data[i][j][1]
            if data[i][j][1] > y_max:
                y_max = data[i][j][1]
            if data[i][j][2] < z_min:
                z_min = data[i][j][2]
            if data[i][j][2] > z_max:
                z_max = data[i][j][2]
            point_data[0, j] = (torch.tensor(data[i][j][0]) - X_MIN) / (X_MAX - X_MIN)
            point_data[1, j] = (torch.tensor(data[i][j][1]) - Y_MIN) / (Y_MAX - Y_MIN)
            point_data[2, j] = (torch.tensor(data[i][j][2]) - Z_MIN) / (Z_MAX - Z_MIN) 
            point_data[3, j] = torch.tensor(data[i][j][3]) / 255.0
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        bounding_boxes.append([center, [x_max - x_min, y_max - y_min, z_max - z_min]])
        input_data.append(point_data)
    if len(input_data) != 0:
        input_data = torch.stack(input_data, dim=0)
    else:
        input_data = None
    return input_data, bounding_boxes
        
def iou_score(bbx1, bbx2):
    x_min = max(bbx1[0][0], bbx2[0][0])
    x_max = min(bbx1[1][0], bbx2[1][0])
    y_min = max(bbx1[0][1], bbx2[0][1])
    y_max = min(bbx1[1][1], bbx2[1][1])
    z_min = max(bbx1[0][2], bbx2[0][2])
    z_max = min(bbx1[1][2], bbx2[1][2])
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min) * max(0, z_max - z_min)
    union = (bbx1[1][0] - bbx1[0][0]) * (bbx1[1][1] - bbx1[0][1]) * (bbx1[1][2] - bbx1[0][2]) \
        + (bbx2[1][0] - bbx2[0][0]) * (bbx2[1][1] - bbx2[0][1]) * (bbx2[1][2] - bbx2[0][2]) - intersection
    return intersection / union