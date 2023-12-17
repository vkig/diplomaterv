import struct
from typing import List
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt


class Point:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.intensity = 0.0


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
