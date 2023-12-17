import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from config import *
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from utils import *

class PointCloudDataset(Dataset):
    def __init__(self, annotation_path: str, points_path, index_path):
        self.annotation_path = annotation_path
        self.points_path = points_path
        self.index_path = index_path
        self.indices = pd.read_csv(index_path, sep=";")["index"]
        self.points = []
        self.targets = []
        self.size = 0
        self.predicted_bounding_boxes = []
        self.ground_truth_bounding_boxes = []
        self.point_clouds = []
        self.object_clouds = []
        self.ious = []
        self.found_objects = 0
        self.not_found_birds = 0
        self.not_found_drones = 0
        self.real_objects = 0
        self.not_found_objects = 0
        self.test = "test" in index_path
        for index in tqdm(self.indices, desc=f"Loading data based on {index_path}", total=len(self.indices)):
            point_csv = pd.read_csv(os.path.join(points_path, f"{index}"), sep=";")
            annotation_csv = pd.read_csv(os.path.join(annotation_path, f"{index}"), sep=";")
            self.parse_one_point(point_csv, annotation_csv)
            
               
    def parse_one_point(self, point_csv, annotation_csv):
        point_xyzi = np.array([[point_csv["x"].iloc[i], point_csv["y"].iloc[i], point_csv["z"].iloc[i], point_csv["intensity"].iloc[i]] for i in range(len(point_csv))])
        self.point_clouds.append(list(map(lambda point: Point(point[0], point[1], point[2], point[3]), point_xyzi)))
        result = DBSCAN(eps=CLUSTERING_EPS, min_samples=POINTS_IN_THE_BOX_THRESHOLD).fit(point_xyzi[:, :3])
        num_of_clusters = len(set(result.labels_))
        if -1 in result.labels_:
            num_of_clusters -= 1
        self.real_objects += len(annotation_csv)
        data = [[] for i in range(num_of_clusters)]
        for i in range(result.labels_.shape[0]):
            if result.labels_[i] != -1:
                data[result.labels_[i]].append(point_xyzi[i])
        found_annotation_indices = []
        for i in range(len(data)):
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
            self.predicted_bounding_boxes.append([[x_min, y_min, z_min], [x_max, y_max, z_max]])
            self.object_clouds.append(len(self.point_clouds) - 1)
            target = -1
            minimum_distance, minimum_distance_i = 100000, -1
            for idx, annotation in annotation_csv.iterrows():
                distance = (annotation["x"] - center[0]) ** 2 + (annotation["y"] - center[1]) ** 2 + (annotation["z"] - center[2]) ** 2
                if distance ** (1 / 2) < CLUSTERING_EPS:
                    if minimum_distance > distance:
                        minimum_distance = distance
                        minimum_distance_i = idx
                if distance < minimum_distance:
                    minimum_distance = distance
            if minimum_distance_i != -1:
                target = 0 if annotation["label"] == "bird" else 1
                annotation = annotation_csv.iloc[minimum_distance_i]
                found_annotation_indices.append(minimum_distance_i)
                self.ground_truth_bounding_boxes.append([[annotation["x"] - annotation["length"] / 2, annotation["y"] - annotation["width"] / 2, \
                    annotation["z"] - annotation["height"] / 2], [annotation["x"] + annotation["length"] / 2 , annotation["y"] + annotation["width"] / 2, \
                        annotation["z"] + annotation["height"] / 2]])
            if target == -1:
                self.ground_truth_bounding_boxes.append(self.predicted_bounding_boxes[-1])
                target = 2
            self.targets.append(target)
            self.points.append(point_data)
            self.size += 1
        self.not_found_objects += len(annotation_csv) - len(set(found_annotation_indices))
        self.found_objects += len(set(found_annotation_indices))
        if annotation_csv["label"].iloc[0] == "bird":
            self.not_found_birds += len(annotation_csv) - len(set(found_annotation_indices))
        else:
            self.not_found_drones += len(annotation_csv) - len(set(found_annotation_indices))

    def __len__(self):
        return self.size
    
    def calculate_iou(self, idx):
        truth_bbx = self.get_ground_truth_bounding_boxes(idx)
        points_in_truth, points_in_predicted = self.get_points_in_bounding_boxes(idx)
        
        x_min = X_MAX
        x_max = X_MIN
        y_min = Y_MAX
        y_max = Y_MIN
        z_min = Z_MAX
        z_max = Z_MIN
        
        for point in points_in_truth:
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
        new_truth_bbx = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        
        number_of_match = 0
        number_of_not_match = 0
        if truth_bbx[0][0] == 0 and truth_bbx[0][1] == 0 and truth_bbx[0][2] == 0 and truth_bbx[1][0] == 0 and truth_bbx[1][1] == 0 and truth_bbx[1][2] == 0:
            return 0
        for point1 in points_in_truth:
            found = False
            for point2 in points_in_predicted:
                if point1.x == point2.x and point1.y == point2.y and point1.z == point2.z and point1.intensity == point2.intensity:
                    number_of_match += 1
                    found = True
                    break
            if not found:
                number_of_not_match += 1
        return iou_score(self.predicted_bounding_boxes[idx], new_truth_bbx)

    def __getitem__(self, idx):
        if self.test:
            return self.points[idx], self.targets[idx], self.predicted_bounding_boxes[idx], self.calculate_iou(idx)
        else:
            return self.points[idx], self.targets[idx], self.predicted_bounding_boxes[idx]
    
    def get_ground_truth_bounding_boxes(self, idx):
        return self.ground_truth_bounding_boxes[idx]
    
    def get_points_in_bounding_boxes(self, idx):
        box = Box()
        box.x_min = self.ground_truth_bounding_boxes[idx][0][0]
        box.y_min = self.ground_truth_bounding_boxes[idx][0][1]
        box.z_min = self.ground_truth_bounding_boxes[idx][0][2]
        box.x_max = self.ground_truth_bounding_boxes[idx][1][0]
        box.y_max = self.ground_truth_bounding_boxes[idx][1][1]
        box.z_max = self.ground_truth_bounding_boxes[idx][1][2]
        points_in_ground_truth = filter_points_in_box(self.point_clouds[self.object_clouds[idx]], box)
        box.x_min = self.predicted_bounding_boxes[idx][0][0]
        box.y_min = self.predicted_bounding_boxes[idx][0][1]
        box.z_min = self.predicted_bounding_boxes[idx][0][2]
        box.x_max = self.predicted_bounding_boxes[idx][1][0]
        box.y_max = self.predicted_bounding_boxes[idx][1][1]
        box.z_max = self.predicted_bounding_boxes[idx][1][2]
        points_in_predicted = filter_points_in_box(self.point_clouds[self.object_clouds[idx]], box)
        return points_in_ground_truth, points_in_predicted


"""
training_data = PointCloudDataset(
    annotation_path="/home/vkig/diplomaterv/database/annotations",
    points_path="/home/vkig/diplomaterv/database/points",
    index_path="/home/vkig/diplomaterv/database/indices/train_indices.csv"
)

points, annotations = training_data.__getitem__(9899)
print(points, annotations)

while(True):
    idx = int(input())
    points, annotations = training_data.__getitem__(idx)
    print(points, annotations)
"""