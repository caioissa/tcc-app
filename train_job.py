import numpy as np

from ml_model import init_model, MODEL_PATH
from ml_service import train


train_set = np.load('data/train.npy', allow_pickle=True)
train_set = train_set/255
with open('data/train-labels.txt', 'r') as f:
    train_labels = np.array([int(c) for c in f.readline()])

model = init_model()
model = train(images=train_set,
              labels=train_labels,
              model=model)

weights = np.array(model.get_weights())
weights.dump(MODEL_PATH)
