import os.path
from copy import deepcopy
from tqdm import tqdm
import rosbag
import yaml
from utils import *
from config import *
from typing import List


def save_annotation_old(annotations, track_idx, label_idx, data_index):
    global annotation_index
    annotation = deepcopy(annotations["tracks"][track_idx]["track"][label_idx])
    annotation.pop("rotation")
    annotation.pop("tags")
    annotation = {
        "id": annotation_index,
        "points_file": str(data_index),
        "annotation": annotation
    }
    all_annotations["annotations"].append(annotation)
    annotation_index += 1


def save_annotation(annotations, track_idx, label_idx, data_index):
    global annotation_index
    annotation = deepcopy(annotations["tracks"][track_idx]["track"][label_idx])
    if not os.path.exists(os.path.join(ANNOTATION_PATH, str(data_index))):
        with open(os.path.join(ANNOTATION_PATH, str(data_index)), 'w') as f:
            f.write("label;x;y;z;length;width;height\n")
    with open(os.path.join(ANNOTATION_PATH, str(data_index)), 'a') as f:
        f.write(f'{annotation["label"]};{annotation["translation"]["x"]};{annotation["translation"]["y"]};{annotation["translation"]["z"]};{annotation["box"]["length"]};{annotation["box"]["width"]};{annotation["box"]["height"]}\n')
    annotation_index += 1


def write_points_to_file(points: List[Point]):
    global data_index
    with open(os.path.join(POINTS_PATH, str(data_index)), 'w') as f:
        f.write("x;y;z;intensity\n")
        for point in points:
            f.write(str(point.x) + ";" + str(point.y) + ";" + str(point.z) + ";" + str(point.intensity) + "\n")


LOG_FILE = "/home/vkig/diplomaterv/database/log.txt"
all_annotations = {"annotations": []}

data_index = 0
annotation_index = 0
MAX_POINTS = 0

for filename in FILENAMES:
    bag = rosbag.Bag(os.path.join(FILTERED_DATA_PATH, filename + ".bag"))
    annotations = yaml.safe_load(open(os.path.join(ANNOTATION_FILES_PATH, filename + ".yaml"), 'r'))

    track_index = 0
    label_index = 0

    for topic, msg, t in tqdm(bag.read_messages(topics=["/filtered_cloud"]),
                              total=bag.get_message_count(topic_filters=["/filtered_cloud"])):
        time_bag_s, time_bag_ns = msg.header.stamp.secs, msg.header.stamp.nsecs
        points = []
        save_points = False
        for i in range(len(annotations["tracks"])):
            for j in range(len(annotations["tracks"][i]["track"])):
                time_annotation_s, time_annotation_ns = time_from_annotation_ns(annotations, i, j)
                plus_threshold = time_annotation_ns + TIME_THRESHOLD
                minus_threshold = time_annotation_ns - TIME_THRESHOLD
                if time_bag_s == time_annotation_s and time_annotation_ns + TIME_THRESHOLD > time_bag_ns > time_annotation_ns - TIME_THRESHOLD:
                    if len(points) == 0:
                        data_array = bytearray(msg.data)
                        for offset in range(0, msg.row_step, msg.point_step):
                            point = parse_point(data_array[offset:offset + msg.point_step])
                            points.append(point)
                    box = parse_annotation(annotations["tracks"][i]["track"][j])
                    points_in_the_box = filter_points_in_box(points, box)
                    if len(points_in_the_box) >= POINTS_IN_THE_BOX_THRESHOLD:
                        if not save_points:
                            save_points = True
                        save_annotation(annotations, i, j, data_index)
        if save_points:
            write_points_to_file(points)
            if len(points) > MAX_POINTS:
                MAX_POINTS = len(points)
            data_index += 1
    bag.close()
    with open("/home/vkig/diplomaterv/database/" + filename + ".yaml", 'w') as f:
        yaml.dump(all_annotations, f)
        del all_annotations
        all_annotations = {"annotations": []}
    with open(LOG_FILE, 'a') as f:
        f.write(filename + " " + str(data_index) + "\n")

print(MAX_POINTS)

