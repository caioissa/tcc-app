import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from ml_model import load_model
from ml_service import test, predict

test_set = np.load('data/test.npy', allow_pickle=True)
test_set = test_set / 255
with open('data/test-labels.txt', 'r') as f:
    test_labels = np.array([int(c) for c in f.readline()])

model = load_model()
score = test(images=test_set,
             labels=test_labels,
             model=model)

print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')

prediction = predict(test_set, model)
prediction = prediction > 0.5
cm = confusion_matrix(test_labels, prediction)
sns.heatmap(cm)
plt.savefig('confusion_matrix.png')
