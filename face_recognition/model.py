from torch import nn
from torch.nn import functional as F


class FaceClassify(nn.Module):
    def __init__(self):
        super(FaceClassify, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        self.drop = nn.Dropout2d()
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x
