import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreBaseModel(nn.Module):
    def __init__(self, input_dim, mid_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.fc1 = nn.Linear(input_dim+1, mid_dim)
        self.fc2 = nn.Linear(mid_dim, mid_dim//2)
        self.fc3 = nn.Linear(mid_dim//2, mid_dim//4)
        self.fc4 = nn.Linear(mid_dim//4, input_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
