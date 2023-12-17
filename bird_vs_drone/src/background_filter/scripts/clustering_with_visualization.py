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
    result = DBSCAN(eps=CLUSTERING_EPS, min_samples=POINTS_IN_THE_BOX_THRESHOLD).fit(points_array)
    #if result.labels_.max() < 3:
    #    continue
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 120)
    ax.set_ylim(-20, 30)
    ax.set_zlim(-6, 15)
    fig.suptitle(str(i))
    colors = {-1: 'b', 0: 'r', 1: 'g', 2: 'y', 3: 'c', 4: 'm', 5: 'k', 6: 'w'}
    for j in range(len(points_array)):
        ax.scatter(points_array[j][0], points_array[j][1], points_array[j][2], c=colors[result.labels_[j]], marker='o')
    ax.set_axis_off()
    plt.show()
            
            
            
    






