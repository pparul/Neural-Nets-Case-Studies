# batch norm
# Basic neural net definition
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# # Convert numpy arrays to tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)

# # Wrap in a Dataset, then DataLoader
# dataset = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# for X_batch, y_batch in dataloader:
#     print(f'Shapes are: {X_batch.shape}, {y_batch.shape}')


class Mymodel(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.ff1 = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout)   # handles train/eval automatically
        self.ff2 = nn.Linear(dout, 1)

    def batchnorm(self, x):
        x_mean = torch.mean(x, axis = 0) 
        x_std = torch.std(x, axis = 0)
        x = (x-x_mean)/(x_std+ 1e-5)
        return x
    
    def forward(self, x):
        x = self.ff1(x)
        x = torch.sigmoid(x)
        x = self.bn(x)
        x = self.ff2(x)
        x = torch.sigmoid(x)
        return x