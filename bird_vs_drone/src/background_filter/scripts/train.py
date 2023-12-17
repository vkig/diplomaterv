import torch
import torch.nn as nn
from model import *
from data_loader import *
import torch.optim as optim
import datetime as dt
from torchsummary import summary

all_data = PointCloudDataset(
    annotation_path=ANNOTATION_PATH,
    points_path=POINTS_PATH,
    index_path=TRAIN_INDEX_FILE,
)

train_dataloader = DataLoader(all_data, batch_size=BATCH_SIZE, shuffle=True)

validation_data = PointCloudDataset(
    annotation_path=ANNOTATION_PATH,
    points_path=POINTS_PATH,
    index_path=VALIDATION_INDEX_FILE,
)

validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

net = BirdNet()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net.to(device)

criterion = F.nll_loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_losses = []
avg_train_losses = []
loss_tmp = []
counter = 0
from_epoch = 0
num_of_epochs = 400

max_validation_accuracy = 0.0

files = os.listdir(SAVE_PATH)
files = list(map(lambda x: int(x.split(".")[0]), files))
files.sort()
last_model_index = files[-1]
current_file_model = last_model_index + 1
accuracies = open(ACCURACY_FILE, 'a')

files = os.listdir(LOSS_FILES)
files = list(map(lambda x: int(x.split(".")[0].split("_")[0]), files))
files.sort()
last_train_index = files[-1]
current_train_index = last_train_index + 1
train_losses_file = open(os.path.join(LOSS_FILES, str(current_train_index) + ".txt"), 'w')
validation_losses_file = open(os.path.join(LOSS_FILES, str(current_train_index) + "_validation.txt"), 'w')

net.train()
train_begin = dt.datetime.now().strftime("%m_%d_%H_%M")
for epoch in tqdm(range(from_epoch, from_epoch + num_of_epochs), total=num_of_epochs, desc="Training..."):
    epoch_loss = 0.0
    items_in_epoch = 0
    for batch_idx, (points, target, bbx) in enumerate(train_dataloader):
        points = points.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        net.train()
        output = net(points)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        items_in_epoch += len(output)
        epoch_loss += loss.item()
        
        train_losses.append(loss.item())
        if len(train_losses) > 8:
            avg_train_losses.append(np.mean(train_losses[-7:]))

        if batch_idx % 10 == 0 and batch_idx != 0:
            pred_choice = output.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, batch_idx, len(train_dataloader), loss.item(), correct.item() / float(output.shape[0])))
    all_corrects = 0
    validation_loss = 0.0
    validation_count = 0
    for j, (points, target, bbx) in enumerate(validation_dataloader):
        points = points.cuda()
        target = target.cuda()
        net.eval()
        pred = net(points)
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        all_corrects += correct.item()
        validation_count += len(pred_choice)
        validation_loss += loss.item()
    if all_corrects / float(validation_count) > max_validation_accuracy and all_corrects / float(validation_count) > 0.87:
        max_validation_accuracy = all_corrects / float(validation_count)
        torch.save(net.state_dict(), os.path.join(SAVE_PATH, str(current_file_model) + ".pt"))
        accuracies.write(str(current_file_model) + "model, epoch: " + str(epoch) + " validation accuracy: " + str(max_validation_accuracy) + " validation loss: " + str(validation_loss / validation_count) + " train loss: " + str(epoch_loss / items_in_epoch) + "\n")
        current_file_model += 1
        print('[%d] %s loss: %f accuracy: %f' % (epoch, 'test', validation_loss / validation_count, max_validation_accuracy))
    train_losses_file.write(str(epoch_loss / items_in_epoch) + "\n")
    validation_losses_file.write(str(validation_loss / validation_count) + "\n")
    # if epoch % 10 == 0:
    #     scheduler.step()           

torch.save(net.state_dict(), os.path.join(SAVE_PATH, str(current_file_model) + ".pt"))

accuracies.close()
validation_losses_file.close()
train_losses_file.close()
