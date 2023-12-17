import os
from model import BirdNet
import torch
from torchsummary import summary
from config import *
import matplotlib.pyplot as plt

net = BirdNet()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net.to(device)

summary(net, INPUT_SHAPE)
print(net)


files = os.listdir(LOSS_FILES)
files = list(map(lambda x: int(x.split(".")[0].split("_")[0]), files))
files.sort()
last_train_index = files[-1]
train_losses_file = open(os.path.join(LOSS_FILES, str(last_train_index) + ".txt"), 'r')
validation_losses_file = open(os.path.join(LOSS_FILES, str(last_train_index) + "_validation.txt"), 'r')

validation_losses = list(map(lambda line: float(line.strip()), validation_losses_file.readlines()))
train_losses = list(map(lambda line: float(line.strip()), train_losses_file.readlines()))

print(min(validation_losses))

plt.title("A veszteségfüggvény értékeinek alakulása az epochok függvényében")
plt.plot(train_losses, label="tanítási veszteség")
plt.plot(validation_losses, label="validációs veszteség")
plt.plot([320, 320], [0.005, 0.018], 'r', label="korai megállás helye")
plt.legend()
plt.xlabel("Epochok")
plt.ylabel("Veszteség értéke")
plt.show()