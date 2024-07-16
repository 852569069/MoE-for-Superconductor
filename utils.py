import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

log=0
if log:
    wandb.init(project='predict model', name='MoE')

# writer = SummaryWriter()

class ResNetBlock(nn.Module):
   def __init__(self, in_channels, out_channels, stride=1):
       super(ResNetBlock, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(out_channels)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(out_channels)

       if in_channels != out_channels or stride != 1:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_channels)
           )
       else:
           self.shortcut = nn.Sequential()

   def forward(self, x):
       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)
       out = self.conv2(out)
       out = self.bn2(out)
       out += self.shortcut(x)
       out = self.relu(out)
       return out


class ResNetBlock_1d(nn.Module):
   def __init__(self, in_channels, out_channels, stride=1):
       super(ResNetBlock_1d, self).__init__()
       self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
       self.bn1 = nn.BatchNorm1d(out_channels)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2 = nn.BatchNorm1d(out_channels)

       if in_channels != out_channels or stride != 1:
           self.shortcut = nn.Sequential(
               nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm1d(out_channels)
           )
       else:
           self.shortcut = nn.Sequential()

   def forward(self, x):
       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)
       out = self.conv2(out)
       out = self.bn2(out)
       out += self.shortcut(x)
       out = self.relu(out)
       return out


class Net2(nn.Module):#物理信息专家
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)

        self.convr_1=ResNetBlock_1d(32,32)
        self.convr_2=ResNetBlock_1d(32,64)
        self.convr_3=ResNetBlock_1d(64,128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, x):
        x=x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x=self.convr_1(x)
        x = self.pool(x)
        x=self.convr_2(x)
        x = self.pool(x)
        x=self.convr_3(x)
        x = x.view(-1, 128 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    







#data load
class Data_base():
    def __init__(self):
        self.data=np.array(pd.read_csv(r'data/data.csv')).astype(np.float32)[:,1:]
        self.scaler = MinMaxScaler()
        self.data[:,86:-1]=self.scaler.fit_transform(self.data[:,86:-1])
        self.element=self.data[:,:86]
        self.element_realnum=np.pad(self.element,((0,0),(0,14)),'constant',constant_values=0)
        self.element_realnum=self.element_realnum.reshape(-1,1,10,10)
        self.physic=self.data[:,86:-1]

    def __getitem__(self, item):
        return self.element[item],self.element_realnum[item],self.physic[item],self.data[item][-1]

    def __len__(self):
        return len(self.data)


batch_size=512
device='cuda:4'


data_set=Data_base()
val_size = int(len(data_set) * 0.2)
train_size = len(data_set) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(data_set, [train_size, val_size],generator=torch.Generator().manual_seed(0))


# create validation data loader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)


import torch.nn.functional as F
def criterion(y_true, y_pred, beta=1.0):
    residual = torch.abs(y_true - y_pred)
    condition = (residual < beta).float()
    loss = condition * 0.5 * residual ** 2 / beta + (1 - condition) * (residual - 0.5 * beta)
    return loss.mean()




