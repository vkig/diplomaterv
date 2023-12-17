import os
import random
import pandas as pd
from config import *

first = 0
last = max(list(map(lambda x: int(x), os.listdir(POINTS_PATH))))

indices = [i for i in range(first, last + 1)]

random.shuffle(indices)
random.shuffle(indices)
random.shuffle(indices)

train_indices = indices[:int(len(indices) * 0.8)]
test_indices = indices[int(len(indices) * 0.8):int(len(indices) * 0.9)]
validation_indices = indices[int(len(indices) * 0.9):]

print(len(train_indices) + len(test_indices) + len(validation_indices))

train_df = pd.DataFrame(train_indices, columns=["index"])
test_df = pd.DataFrame(test_indices, columns=["index"])
validation_df = pd.DataFrame(validation_indices, columns=["index"])

train_df.to_csv(os.path.join(INDEX_PATH, "train_indices.csv"), sep=";", index=False)
test_df.to_csv(os.path.join(INDEX_PATH, "test_indices.csv"), sep=";", index=False)
validation_df.to_csv(os.path.join(INDEX_PATH, "validation_indices.csv"), sep=";", index=False)