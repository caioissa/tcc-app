import os

from PIL import Image
import numpy as np

for n in ('train', 'test'):
    array = np.array([np.array(Image.open(f'images/{n}/{image_path}').convert('L'))
                      for image_path in os.listdir(f'images/{n}')])
    labels = ['1' if f.startswith('t') else '0' for f in os.listdir(f'images/{n}')]
    array.dump(f'{n}.npy')
    with open(f'{n}-labels.txt', 'w') as f:
        f.writelines(labels)
