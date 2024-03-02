import torch.nn as nn
import torch.nn.functional as F

class CNNLeaky(nn.Module):
    def __init__(self, output_size):
        super(CNNLeaky, self).__init__()

        # 11 x 11 -> 10 x 10
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=2)
        # 10 x 10 -> 9 x 9
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        # 9 x 9 -> 8 x 8
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        # 8 x 8 -> 7 x 7
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        # 7 x 7 -> 7 x 7
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        # 512 * 7 * 7 -> 25088
        self.fc1 = nn.Linear(25088, 512)

        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # 11 x 11 -> 10 x 10
        x = F.leaky_relu(self.conv1(x))
        # 10 x 10 -> 9 x 9
        x = F.leaky_relu(self.conv2(x))
        # 9 x 9 -> 8 x 8
        x = F.leaky_relu(self.conv3(x))
        # 8 x 8 -> 7 x 7
        x = F.leaky_relu(self.conv4(x))
        # 7 x 7 -> 7 x 7
        x = F.leaky_relu(self.conv5(x))
        
        x = x.view(-1, 25088)

        x = F.leaky_relu(self.fc1(x))
        # don't relu the last layer
        x = self.fc2(x)

        return x