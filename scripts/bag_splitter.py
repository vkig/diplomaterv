import math
import os.path
from tqdm import tqdm
import rosbag
from config import *

FILENAME = "m_hid_3_3.bag"

bag = rosbag.Bag(os.path.join(FILTERED_DATA_PATH, FILENAME))

time_length = bag.get_end_time() - bag.get_start_time()

number_of_parts = math.ceil(time_length / 60.0)

out_bags_filenames = [FILENAME.split('.')[0] + "_" + str(i) + ".bag" for i in range(1, number_of_parts + 1)]
out_bags = [rosbag.Bag(os.path.join(FILTERED_DATA_PATH, filename), 'w') for filename in out_bags_filenames]

background_model_topic, background_model_msg = BACKGROUND_MODEL_TOPIC, 0
for topic, msg, t in bag.read_messages(topics=[BACKGROUND_MODEL_TOPIC]):
    background_model_topic, background_model_msg, _ = topic, msg, t

bounding_box_topic, bounding_box_msg = BOUNDING_BOX_TOPIC, 0
for topic, msg, t in bag.read_messages(topics=[BOUNDING_BOX_TOPIC]):
    bounding_box_topic, bounding_box_msg, _ = topic, msg, t
# map(lambda x: x.write(bounding_box_topic, bounding_box_msg, bounding_box_time), out_bags)

first_time = [True for i in range(number_of_parts)]
for topic, msg, t in tqdm(bag.read_messages(topics=[FILTERED_CLOUD_TOPIC, CAMERA_TOPIC]), total=bag.get_message_count() - 2):
    index = int((t.to_sec() - bag.get_start_time()) // 60)
    if first_time[index]:
        first_time[index] = False
        bounding_box_msg.header.stamp = msg.header.stamp
        background_model_msg.header.stamp = msg.header.stamp
        out_bags[index].write(bounding_box_topic, bounding_box_msg, t)
        out_bags[index].write(background_model_topic, background_model_msg, t)
    out_bags[index].write(topic, msg, t)
    out_bags[index].flush()

for i in range(len(out_bags)):
    out_bags[i].close()
bag.close()
