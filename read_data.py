import torch
from scipy.io import loadmat

data = loadmat("NumStim_7to28_100x100_TrainSet.mat")
print(data.keys())


# numpy array of (10000, 21970)
print(type(data['D']), data['D'].shape)

raw = data['D']   # shape (10000, 21970)

imgs = torch.tensor(raw.T, dtype=torch.float32)
imgs_reshaped = imgs.view(-1, 100, 100)

import matplotlib.pyplot as plt

plt.imshow(imgs_reshaped[0], cmap="gray")
plt.show()

