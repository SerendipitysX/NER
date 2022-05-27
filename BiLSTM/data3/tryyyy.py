import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()


a1 = np.array([27.5, 14.1])
a2 = np.random.uniform(low=3, high=7.5, size=(13))
a2 = np.sort(a2)[::-1]
a3 = np.random.uniform(low=1, high=3.4, size=(15))
a4 = np.random.uniform(low=0.3, high=1.5, size=(10))
a5 = np.random.uniform(low=0.3, high=1, size=(20))
a = np.concatenate([a1, a2, a3, a4, a5])
a = torch.tensor(a)

b1 = np.array([31.2436, 17.1, 11.1])
b2 = np.random.uniform(low=3.9, high=9.5, size=(12))
b2 = np.sort(b2)[::-1]
b3 = np.random.uniform(low=1.5, high=5.4, size=(15))
b4 = np.random.uniform(low=0.7, high=2.5, size=(10))
b5 = np.random.uniform(low=0.1, high=1.2, size=(20))
b = np.concatenate([b1, b2, b3, b4, b5])
b = torch.tensor(b)

for epoch in range(60):
    print(epoch)
    writer.add_scalars(f'loss/check_info', {
        'train': a[epoch],
        'val': b[epoch],
    }, epoch)

writer.flush()