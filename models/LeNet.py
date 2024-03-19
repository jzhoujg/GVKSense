import torch
import torch.nn as nn
import torch.nn.functional as F

class NTU_Fi_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(NTU_Fi_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (3,114,500)
            nn.Conv2d(3,32,(15,23),stride=9),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=(1,3)),
            nn.ReLU(True),
            nn.Conv2d(64,96,(7,3),stride=(1,3)),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*6,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*6)
        out = self.fc(x)
        return out

class UT_HAR_LeNet(nn.Module):
    def __init__(self):
        super(UT_HAR_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out