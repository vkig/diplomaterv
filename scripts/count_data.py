import math
import os
import pandas as pd
from utils import *
from config import *
from tqdm import tqdm


def count_annotations():
    bird = 0
    drone = 0
    for filename in os.listdir(ANNOTATION_PATH):
        file = pd.read_csv(os.path.join(ANNOTATION_PATH, filename), sep=";")
        if file["label"][0] == "bird":
            bird += len(file["label"])
        else:
            drone += 1
    print(f"bird: {bird}, drone: {drone}")


def point_stats():
    drone_min = 24000
    drone_max = 0
    bird_min = 24000
    bird_max = 0
    drone_avg = 0
    bird_avg = 0
    drone = 0
    bird = 0
    points_on_drone_max = 0
    points_on_bird_max = 0
    points_on_drone_avg = 0
    points_on_bird_avg = 0
    max_distance_between_points_on_drone = 0
    max_distance_between_points_on_bird = 0
    average_distance_between_points_on_drone = 0
    average_distance_between_points_on_bird = 0
    maximum_distance_to_nearest_point_on_drone = 0
    maximum_distance_to_nearest_point_on_bird = 0
    distance_divisor_drone = 0
    distance_divisor_bird = 0
    drone_length_max = 0
    drone_length_min = 100
    drone_width_max = 0
    drone_width_min = 100
    drone_height_max = 0
    drone_height_min = 100
    bird_length_max = 0
    bird_length_min = 100
    bird_width_max = 0
    bird_width_min = 100
    bird_height_max = 0
    bird_height_min = 100
    for filename in tqdm(range(DATABASE_SIZE)):
        points_csv = pd.read_csv(os.path.join(POINTS_PATH, str(filename)), sep=";")
        annotation_csv = pd.read_csv(os.path.join(ANNOTATION_PATH, str(filename)), sep=";")
        points = parse_points_from_database(points_csv)
        for _, annotation in annotation_csv.iterrows():
            box = parse_box_from_database(annotation)
            points_in_the_box = filter_points_in_box(points, box)
            if annotation_csv["label"][0] == "bird":
                if annotation["length"] > bird_length_max:
                    bird_length_max = annotation["length"]
                if annotation["length"] < bird_length_min:
                    bird_length_min = annotation["length"]
                if annotation["width"] > bird_width_max:
                    bird_width_max = annotation["width"]
                if annotation["width"] < bird_width_min:
                    bird_width_min = annotation["width"]
                if annotation["height"] > bird_height_max:
                    bird_height_max = annotation["height"]
                if annotation["height"] < bird_height_min:
                    bird_height_min = annotation["height"]
                if len(points_in_the_box) > points_on_bird_max:
                    points_on_bird_max = len(points_in_the_box)
                points_on_bird_avg += len(points_in_the_box)
                for i in range(len(points_in_the_box) - 1):
                    min_distance = 100
                    for j in range(i + 1, len(points_in_the_box)):
                        distance = math.sqrt((points_in_the_box[i].x - points_in_the_box[j].x) ** 2 +
                                             (points_in_the_box[i].y - points_in_the_box[j].y) ** 2 +
                                             (points_in_the_box[i].z - points_in_the_box[j].z) ** 2)
                        average_distance_between_points_on_bird += distance
                        distance_divisor_bird += 1
                        if distance < min_distance:
                            min_distance = distance
                        if distance > max_distance_between_points_on_bird:
                            # plot_points_with_annotations(points, annotation_csv, filename)
                            max_distance_between_points_on_bird = distance
                    if min_distance > maximum_distance_to_nearest_point_on_bird:
                        maximum_distance_to_nearest_point_on_bird = min_distance
                        # plot_points_with_annotations(points, annotation_csv, str(filename), maximum_distance_to_nearest_point_on_bird, "bird")

            else:
                if annotation["length"] > drone_length_max:
                    drone_length_max = annotation["length"]
                if annotation["length"] < drone_length_min:
                    drone_length_min = annotation["length"]
                if annotation["width"] > drone_width_max:
                    drone_width_max = annotation["width"]
                if annotation["width"] < drone_width_min:
                    drone_width_min = annotation["width"]
                if annotation["height"] > drone_height_max:
                    drone_height_max = annotation["height"]
                if annotation["height"] < drone_height_min:
                    drone_height_min = annotation["height"]
                if len(points_in_the_box) > points_on_drone_max:
                    points_on_drone_max = len(points_in_the_box)
                points_on_drone_avg += len(points_in_the_box)
                for i in range(len(points_in_the_box) - 1):
                    min_distance = 100
                    for j in range(i + 1, len(points_in_the_box)):
                        distance = math.sqrt((points_in_the_box[i].x - points_in_the_box[j].x) ** 2 +
                                             (points_in_the_box[i].y - points_in_the_box[j].y) ** 2 +
                                             (points_in_the_box[i].z - points_in_the_box[j].z) ** 2)
                        average_distance_between_points_on_drone += distance
                        distance_divisor_drone += 1
                        if distance < min_distance:
                            min_distance = distance
                        if distance > max_distance_between_points_on_drone:
                            # plot_points_with_annotations(points, annotation_csv, filename)
                            max_distance_between_points_on_drone = distance
                    if min_distance > maximum_distance_to_nearest_point_on_drone:
                        maximum_distance_to_nearest_point_on_drone = min_distance
                        # plot_points_with_annotations(points, annotation_csv, str(filename), maximum_distance_to_nearest_point_on_drone, "drone")
        if annotation_csv["label"][0] == "bird":
            if len(points_csv["x"]) < bird_min:
                bird_min = len(points_csv)
            if len(points_csv["x"]) > bird_max:
                bird_max = len(points_csv)
            bird_avg += len(points_csv["x"])
            bird += len(annotation_csv["label"])
        else:
            if len(points_csv["x"]) < drone_min:
                drone_min = len(points_csv)
            if len(points_csv["x"]) > drone_max:
                drone_max = len(points_csv)
            drone_avg += len(points_csv["x"])
            drone += len(annotation_csv["label"])
    print("bird", bird)
    print("drone", drone)
    print("bird_min", bird_min)
    print("bird_max", bird_max)
    print("drone_min", drone_min)
    print("drone_max", drone_max)
    print("bird_avg", bird_avg / bird)
    print("drone_avg", drone_avg / drone)
    print("points_on_bird_max", points_on_bird_max)
    print("points_on_drone_max", points_on_drone_max)
    print("points_on_bird_avg", points_on_bird_avg / bird)
    print("points_on_drone_avg", points_on_drone_avg / drone)
    print("max_distance_between_points_on_bird", max_distance_between_points_on_bird)
    print("max_distance_between_points_on_drone", max_distance_between_points_on_drone)
    print("average_distance_between_points_on_bird", average_distance_between_points_on_bird / distance_divisor_bird)
    print("average_distance_between_points_on_drone", average_distance_between_points_on_drone / distance_divisor_drone)
    print("drone_length_max", drone_length_max)
    print("drone_length_min", drone_length_min)
    print("drone_width_max", drone_width_max)
    print("drone_width_min", drone_width_min)
    print("drone_height_max", drone_height_max)
    print("drone_height_min", drone_height_min)
    print("bird_length_max", bird_length_max)
    print("bird_length_min", bird_length_min)
    print("bird_width_max", bird_width_max)
    print("bird_width_min", bird_width_min)
    print("bird_height_max", bird_height_max)
    print("bird_height_min", bird_height_min)
    print("maximum_distance_to_nearest_point_on_drone", maximum_distance_to_nearest_point_on_drone)
    print("maximum_distance_to_nearest_point_on_bird", maximum_distance_to_nearest_point_on_bird)


point_stats()
