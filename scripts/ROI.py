import os.path
from tqdm import tqdm
import rosbag
from utils import *
from config import *
import pandas as pd

bird = 0
drone = 0
for i in range(5219):
    csv = pd.read_csv(os.path.join(ANNOTATION_PATH, str(i)), sep=";")
    if csv["label"][0] == "bird":
        bird += len(csv)
    else:
        drone += 1
        
print(bird)
print(drone)
quit()


for filename in FILENAMES:
    bag = rosbag.Bag(os.path.join(FILTERED_DATA_PATH, filename + ".bag"))

    x_min = 1000
    x_max = -1000
    y_min = 1000
    y_max = -1000
    z_min = 1000
    z_max = -1000

    for topic, msg, t in tqdm(bag.read_messages(topics=["/filtered_cloud"]),
                              total=bag.get_message_count(topic_filters=["/filtered_cloud"])):
        points = []
        data_array = bytearray(msg.data)
        for offset in range(0, msg.row_step, msg.point_step):
            point = parse_point(data_array[offset:offset + msg.point_step])
            points.append(point)
        for point in points:
            if point.x < x_min:
                x_min = point.x
            if point.x > x_max:
                x_max = point.x
            if point.y < y_min:
                y_min = point.y
            if point.y > y_max:
                y_max = point.y
            if point.z < z_min:
                z_min = point.z
            if point.z > z_max:
                z_max = point.z

    print(filename)
    print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, z_min: {z_min}, z_max: {z_max}")
    bag.close()
