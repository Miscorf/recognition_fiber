import torch.nn as nn
import torch.nn.functional as F

class model_linear(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv1d(50,1024,3)
        self.fc1 = nn.Linear(10,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(3)
        self.pool = nn.MaxPool1d(2,2)

        
    def forward(self, x):
        # x = self.conv1(x)
        # x = F.tanh(x)
        # x = self.pool(x)
        # # tanh 较好
        # # x = x.view(-1, (28*28)//(4*4)*8)
        # x = x.view(1,-1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = F.dropout(x, p=0.4,training=self.training)
        
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.4,training=self.training)
        
        x = self.fc3(x)
        # x = sfmax(x)
        return x
