import os.path
import random
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *
from utils import *


def filter_object_points(points, box):
    object_points = []
    for i in range(len(points)):
        if box.x_min <= points.iloc[i]["x"] <= box.x_max and \
                box.y_min <= points.iloc[i]["y"] <= box.y_max and \
                box.z_min <= points.iloc[i]["z"] <= box.z_max:
            object_points.append(i)
    return object_points


def rotate(points, annotations):
    new_points = deepcopy(points)
    new_annotations = deepcopy(annotations)
    theta_deg = random.randrange(0, 360)
    theta = np.radians(theta_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_z = [
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ]
    rotation_z = np.array(rotation_z)

    for j in range(len(annotations)):
        center = np.array([annotations.iloc[j]["x"], annotations.iloc[j]["y"], annotations.iloc[j]["z"]])
        box = parse_box_from_database(annotations.iloc[j])
        object_points = filter_object_points(points, box)
        for index in object_points:
            vector = np.array([points.iloc[index]["x"], points.iloc[index]["y"], points.iloc[index]["z"]])
            vector = vector - center
            vector = np.dot(vector, rotation_z)
            vector = vector + center
            new_points.loc[index, "x"] = vector[0]
            new_points.loc[index, "y"] = vector[1]
            new_points.loc[index, "z"] = vector[2]
            if box.x_min > vector[0]:
                box.x_min = vector[0]
            if box.x_max < vector[0]:
                box.x_max = vector[0]
            if box.y_min > vector[1]:
                box.y_min = vector[1]
            if box.y_max < vector[1]:
                box.y_max = vector[1]
            if box.z_min > vector[2]:
                box.z_min = vector[2]
            if box.z_max < vector[2]:
                box.z_max = vector[2]
        new_annotations.loc[j, "x"] = (box.x_min + box.x_max) / 2
        new_annotations.loc[j, "y"] = (box.y_min + box.y_max) / 2
        new_annotations.loc[j, "z"] = (box.z_min + box.z_max) / 2
        new_annotations.loc[j, "length"] = box.x_max - box.x_min
        new_annotations.loc[j, "width"] = box.y_max - box.y_min
        new_annotations.loc[j, "height"] = box.z_max - box.z_min
    return new_points, new_annotations


def shift(points, annotations):
    new_points = deepcopy(points)
    new_annotations = deepcopy(annotations)
    for i in range(len(annotations)):
        box = parse_box_from_database(annotations.iloc[i])
        new_box = Box()
        new_box.x_min = box.x_min
        new_box.x_max = box.x_max
        new_box.y_min = box.y_min
        new_box.y_max = box.y_max
        new_box.z_min = box.z_min
        new_box.z_max = box.z_max
        points_in_new = filter_object_points(new_points, new_box)
        x_shift = 0
        y_shift = 0
        z_shift = 0
        while len(points_in_new) != 0:
            x_shift = random.randrange(-int(min(abs(box.x_min - X_MIN), 10)), int(min(abs(box.x_max - X_MAX), 10)))
            y_shift = random.randrange(-int(min(abs(box.y_min - Y_MIN), 10)), int(min(abs(box.y_max - Y_MAX), 10)))
            z_shift = random.randrange(-int(min(abs(box.z_min - Z_MIN), 10)), int(min(abs(box.z_max - Z_MAX), 10)))
            new_box.x_min = box.x_min + x_shift
            new_box.x_max = box.x_max + x_shift
            new_box.y_min = box.y_min + y_shift
            new_box.y_max = box.y_max + y_shift
            new_box.z_min = box.z_min + z_shift
            new_box.z_max = box.z_max + z_shift
            points_in_new = filter_object_points(new_points, new_box)
        object_points = filter_object_points(points, box)
        for index in object_points:
            new_points.loc[index, "x"] = points.iloc[index]["x"] + x_shift
            new_points.loc[index, "y"] = points.iloc[index]["y"] + y_shift
            new_points.loc[index, "z"] = points.iloc[index]["z"] + z_shift
        new_annotations.loc[i, "x"] = annotations.iloc[i]["x"] + x_shift
        new_annotations.loc[i, "y"] = annotations.iloc[i]["y"] + y_shift
        new_annotations.loc[i, "z"] = annotations.iloc[i]["z"] + z_shift
    return new_points, new_annotations


def save_new(points, annotations, index):
    points.to_csv(os.path.join(POINTS_PATH, str(index)), sep=";", index=False)
    annotations.to_csv(os.path.join(ANNOTATION_PATH, str(index)), sep=";", index=False)

LOG_AUGMENTATION = "/home/vkig/diplomaterv/database/augmentation_log.txt"
f = open(LOG_AUGMENTATION, "w")
save_index = 5219
filenames = os.listdir(POINTS_PATH)

for index in tqdm(range(5219)):
    filename = str(index)
    points = pd.read_csv(os.path.join(POINTS_PATH, filename), sep=";")
    annotations = pd.read_csv(os.path.join(ANNOTATION_PATH, filename), sep=";")

    if annotations["label"][0] == "bird":
        for i in range(2):
            new_points, new_annotations = rotate(points, annotations)
            save_new(new_points, new_annotations, save_index)
            f.write(str(save_index) + ":" + str(index) + "\n")
            save_index += 1

            new_points, new_annotations = shift(points, annotations)
            save_new(new_points, new_annotations, save_index)
            f.write(str(save_index) + ":" + str(index) + "\n")
            save_index += 1

        for i in range(2):
            new_points, new_annotations = rotate(points, annotations)
            new_points, new_annotations = shift(new_points, new_annotations)
            save_new(new_points, new_annotations, save_index)
            f.write(str(save_index) + ":" + str(index) + "\n")
            save_index += 1
    else:
        new_points, new_annotations = shift(points, annotations)
        save_new(new_points, new_annotations, save_index)
        f.write(str(save_index) + ":" + str(index) + "\n")
        save_index += 1

        new_points, new_annotations = rotate(points, annotations)
        new_points, new_annotations = shift(new_points, new_annotations)
        save_new(new_points, new_annotations, save_index)
        f.write(str(save_index) + ":" + str(index) + "\n")
        save_index += 1
