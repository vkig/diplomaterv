from data_loader import *

all_data = PointCloudDataset(
    annotation_path=ANNOTATION_PATH,
    points_path=POINTS_PATH,
    index_path=ALL_INDEX_FILE,
)

print("found_objects: ", all_data.found_objects)
print("real_objects: ", all_data.real_objects)
print("not_found_objects: ", all_data.not_found_objects)
print("not_found_birds: ", all_data.not_found_birds)
print("not_found_drones: ", all_data.not_found_drones)
print("extra objects: ", all_data.targets.count(2))

avg_iou = 0.0
count = 0
matches = []
not_matches = []
avg_matches = 0.0
avg_not_matches = 0.0
for i in range(all_data.__len__()):
    points, target, bbx = all_data.__getitem__(i)
    truth_bbx = all_data.get_ground_truth_bounding_boxes(i)
    points_in_truth, points_in_predicted = all_data.get_points_in_bounding_boxes(i)
    
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
        continue
    for point1 in points_in_truth:
        found = False
        for point2 in points_in_predicted:
            if point1.x == point2.x and point1.y == point2.y and point1.z == point2.z and point1.intensity == point2.intensity:
                number_of_match += 1
                found = True
                break
        if not found:
            number_of_not_match += 1
    matches.append(number_of_match)
    not_matches.append(number_of_not_match)
    avg_matches += number_of_match / len(points_in_truth)
    avg_not_matches += number_of_not_match / len(points_in_truth)
    count += 1
    avg_iou += iou_score(bbx, new_truth_bbx)
avg_iou /= count
print("average iou: ", avg_iou)
print("average matches: ", avg_matches / count)
print("average not matches: ", avg_not_matches / count)