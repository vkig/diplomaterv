import torch
from utils import iou_score
from model import *
from data_loader import *
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

test_data = PointCloudDataset(
    annotation_path = ANNOTATION_PATH,
    points_path = POINTS_PATH,
    index_path = TEST_INDEX_FILE
)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

net = BirdNet()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net.to(device)

files = os.listdir(SAVE_PATH)
files = list(map(lambda x: int(x.split(".")[0]), files))
files.sort()
last_file = files[-1]

net.load_state_dict(torch.load(os.path.join(SAVE_PATH, "80" + ".pt"), map_location=device))

net.eval()
total = 0
test_loss = 0.0

confusion_matrix = torch.zeros(3, 3)
iou_confusion_matrix = [torch.zeros(3, 2) for i in range(9)]

criterion = F.nll_loss
average_accuracy = [0.0, 0.0, 0.0]
object_count = [0, 0, 0]

y_true = []
bird_scores = []
drone_scores = []
other_scores = []
scores = []

with torch.no_grad():
    correct = 0
    average_loss = 0.0
    for j, (points, target, bbx, iou) in enumerate(test_dataloader):
        points = points.to(device)
        target = target.to(device)
        net.eval()
        pred = net(points)
        loss = F.nll_loss(pred, target)
        average_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum()
        
        if torch.sum(target == 0).item() != 0:
            average_accuracy[0] += torch.sum((pred_choice == 0) & (target == 0)).item()
            object_count[0] += torch.sum(target == 0).item()
        if torch.sum(target == 1).item() != 0:
            average_accuracy[1] += torch.sum((pred_choice == 1) & (target == 1)).item()
            object_count[1] += torch.sum(target == 1).item()
        if torch.sum(target == 2).item() != 0:
            average_accuracy[2] += torch.sum((pred_choice == 2) & (target == 2)).item()
            object_count[2] += torch.sum(target == 2).item()
        for i in range(len(target)):
            confusion_matrix[target[i]][pred_choice[i]] += 1
            for j in range(1, 10):
                if iou[i] > 0.1 * j and target[i] == pred_choice[i]:
                    iou_confusion_matrix[j - 1][target[i]][0] += 1
                else:
                    iou_confusion_matrix[j - 1][target[i]][1] += 1
            y_true.append(target[i].cpu().item())
            bird_scores.append(torch.exp(pred[i, 0].cpu()).item())
            drone_scores.append(torch.exp(pred[i, 1].cpu()).item())
            other_scores.append(torch.exp(pred[i, 2].cpu()).item())
            scores.append(torch.exp(pred[i].cpu()).numpy().tolist())
    for j in range(9):
        ap_bird = iou_confusion_matrix[j][0][0] / (iou_confusion_matrix[j][0][0] + iou_confusion_matrix[j][0][1])
        ap_drone = iou_confusion_matrix[j][1][0] / (iou_confusion_matrix[j][1][0] + iou_confusion_matrix[j][1][1])
        ap_other = iou_confusion_matrix[j][2][0] / (iou_confusion_matrix[j][2][0] + iou_confusion_matrix[j][2][1])
        mAP = (ap_bird + ap_drone + ap_other) / 3
        print("mAP @", 0.1 * (j + 1), " IoU: ", mAP)
                
    print('average loss: %f average accuracy: %f' % (average_loss / float(len(test_dataloader)), correct.item()/float(len(test_data))))
    print('average accuracy for bird: %f' % (average_accuracy[0] / float(object_count[0])))
    print('average accuracy for drone: %f' % (average_accuracy[1] / float(object_count[1])))
    print('average accuracy for other: %f' % (average_accuracy[2] / float(object_count[2])))
    
    accuracy = 0.0
    for i in range(3):
        accuracy += confusion_matrix[i][i].item()
    accuracy /= torch.sum(confusion_matrix).item()
    print("accuracy: ", accuracy)
    bird_precision = confusion_matrix[0][0].item() / torch.sum(confusion_matrix[:, 0]).item()
    drone_precision = confusion_matrix[1][1].item() / torch.sum(confusion_matrix[:, 1]).item()
    other_precision = confusion_matrix[2][2].item() / torch.sum(confusion_matrix[:, 2]).item()
    print("bird precision: ", bird_precision)
    print("drone precision: ", drone_precision)
    print("other precision: ", other_precision)
    bird_recall = confusion_matrix[0][0].item() / torch.sum(confusion_matrix[0, :]).item()
    drone_recall = confusion_matrix[1][1].item() / torch.sum(confusion_matrix[1, :]).item()
    other_recall = confusion_matrix[2][2].item() / torch.sum(confusion_matrix[2, :]).item()
    print("bird recall: ", bird_recall)
    print("drone recall: ", drone_recall)
    print("other recall: ", other_recall)
    bird_f1 = 2 * bird_precision * bird_recall / (bird_precision + bird_recall)
    drone_f1 = 2 * drone_precision * drone_recall / (drone_precision + drone_recall)
    other_f1 = 2 * other_precision * other_recall / (other_precision + other_recall)
    print("bird f1: ", bird_f1)
    print("drone f1: ", drone_f1)
    print("other f1: ", other_f1)
    
    roc_curve_bird = roc_curve(y_true, bird_scores, pos_label=0)
    roc_curve_drone = roc_curve(y_true, drone_scores, pos_label=1)
    roc_curve_other = roc_curve(y_true, other_scores, pos_label=2)
    
    plt.plot(roc_curve_bird[0], roc_curve_bird[1], label="madár")
    plt.plot(roc_curve_drone[0], roc_curve_drone[1], label="drón")
    plt.plot(roc_curve_other[0], roc_curve_other[1], label="egyéb")
    plt.legend()
    plt.title("ROC görbe")
    plt.xlabel("Hamis pozitív arány")
    plt.ylabel("Valós pozitív arány")
    plt.show()
    
    roc_auc = roc_auc_score(y_true, scores, multi_class="ovr")
    print("roc auc score (ovr): ", roc_auc)
    
    roc_auc = roc_auc_score(y_true, scores, multi_class="ovo")
    print("roc auc score (ovo): ", roc_auc)
    
    ap = average_precision_score(y_true, scores)
    print("average precision score: ", ap)
    
    bird_ap = average_precision_score(np.array(y_true) == 0, bird_scores)
    drone_ap = average_precision_score(np.array(y_true) == 1, drone_scores)
    other_ap = average_precision_score(np.array(y_true) == 2, other_scores)
    print("bird average precision score: ", bird_ap)
    print("drone average precision score: ", drone_ap)
    print("other average precision score: ", other_ap)
    print("map", (bird_ap + drone_ap + other_ap) / 3)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix.numpy(), display_labels=["madár", "drón", "egyéb"])
    disp.plot()
    plt.title("Igazságmátrix")
    plt.xlabel("Prediktált címke")
    plt.ylabel("Valós címke")
    plt.show()

avg_iou = 0.0
count = 0
matches = []
not_matches = []
avg_matches = 0.0
avg_not_matches = 0.0
min_iou = 1.0
for i in range(test_data.__len__()):
    points, target, bbx, _ = test_data.__getitem__(i)
    truth_bbx = test_data.get_ground_truth_bounding_boxes(i)
    points_in_truth, points_in_predicted = test_data.get_points_in_bounding_boxes(i)
    
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
    iou = iou_score(bbx, new_truth_bbx)
    if iou < min_iou and iou != 0.0:
        min_iou = iou
    avg_iou += iou
avg_iou /= count
print("average iou: ", avg_iou)
print("average matches: ", avg_matches / count)
print("average not matches: ", avg_not_matches / count)
print("min iou: ", min_iou)
    