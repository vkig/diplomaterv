import time

from sklearn.cluster import DBSCAN
from config import *
from utils import *
import pandas as pd
import os

last = max(list(map(lambda x: int(x), os.listdir(POINTS_PATH))))

for i in range(last + 1):
    points_csv = pd.read_csv(os.path.join(POINTS_PATH, str(i)), sep=";")
    points = parse_points_from_database(points_csv)
    points_array = []
    for point in points:
        points_array.append([point.x, point.y, point.z])
    tick = time.time_ns()
    result = DBSCAN(eps=CLUSTERING_EPS, min_samples=POINTS_IN_THE_BOX_THRESHOLD).fit(points_array)
    tock = time.time_ns()
    print("Time: " + str(tock - tick))
    if result.labels_.max() > 2:
        annotation_csv = pd.read_csv(os.path.join(ANNOTATION_PATH, str(i)), sep=";")
        boxes = []
        for annotation in annotation_csv.iterrows():
            boxes.append(parse_box_from_database(annotation[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 120)
        ax.set_ylim(-20, 30)
        ax.set_zlim(-6, 15)
        colors = {-1: 'b', 0: 'r', 1: 'g', 2: 'y', 3: 'c', 4: 'm', 5: 'k', 6: 'w'}

        for j in range(len(points)):
            ax.scatter(points[j].x, points[j].y, points[j].z, c=colors[result.labels_[j]], marker='o')
        for box in boxes:
            ax.scatter(box.x_min, box.y_min, box.z_min, c='r', marker='^')
            ax.scatter(box.x_max, box.y_max, box.z_max, c='r', marker='^')
        ax.set_title(str(i) + ", " + str(len(boxes)))
        ax.set_axis_off()
        plt.show()




