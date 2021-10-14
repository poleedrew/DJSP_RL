import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline_DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(Baseline_DQN, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        self.fc5 = nn.Linear(30, action_space)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        Q_value = self.fc5(x)
        return Q_value


class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.conv_space, self.vector_sapce = state_space
        self.action_space = action_space
        self.conv1 = nn.Conv2d(in_channels = self.conv_space[0], out_channels = 3, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding=1)
        self.midSize = self.__pre_calculate()
        # print("Conv flatten size: ", self.midSize)
        self.fc1 = nn.Linear(self.midSize + self.vector_sapce, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, self.action_space)

    def forward(self, s1, s2):
        x = torch.relu(self.conv1(s1))
        x = torch.relu(self.conv2(x)).view(x.size()[0], -1)
        # print(x.size())
        # print(s2.size())
        x = torch.cat((x, s2), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        Q_value = self.fc3(x)
        return Q_value
    
    def __pre_calculate(self):
        x = torch.unsqueeze(torch.rand(self.conv_space), 0)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            return x.view(1, -1).size()[1]
        

