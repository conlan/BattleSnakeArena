import torch.nn as nn
import torch.nn.functional as F

class CNNLeaky(nn.Module):
    def __init__(self, output_size):
        super(CNNLeaky, self).__init__()

        # # 67 x 67 -> 16 x 16
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=4)      
        # # 16 x 16 -> 7 x 7
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # # 7 x 7 -> 5 x 5
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # # 64 x 5 x 5 -> 512
        # self.fc1 = nn.Linear(1600, 512)
        # self.fc2 = nn.Linear(512, output_size)

        # (67-3)/2 + 1 -> 33 x 33
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        # (33-3)/2 + 1 -> 16 x 16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        # (16-3)/1 + 1 -> 14 x 14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # (14-3)/1 + 1 -> 12 x 12
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        # 256 * 12 * 12 -> 4608
        self.fc1 = nn.Linear(36864, 512)

        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # 67 x 67 -> 33 x 33
        x = F.leaky_relu(self.conv1(x))
        # 33 x 33 -> 16 x 16
        x = F.leaky_relu(self.conv2(x))
        # 16 x 16 -> 14 x 14
        x = F.leaky_relu(self.conv3(x))
        # 14 x 14 -> 12 x 12
        x = F.leaky_relu(self.conv4(x))
        # 256 * 12 * 12 -> 4608
        x = x.view(-1, 36864)
        # 4608 -> 512
        x = F.leaky_relu(self.fc1(x))
        # don't relu the last layer
        x = self.fc2(x)

        # # 67 x 67 -> 16 x 16
        # x = F.leaky_relu(self.conv1(x))        
        # # 16 x 16 -> 7 x 7
        # x = F.leaky_relu(self.conv2(x))
        # # 7 x 7 -> 5 x 5
        # x = F.leaky_relu(self.conv3(x))
        # # 64 x 5 x 5 -> 1600
        # x = x.view(-1, 64 * 5 * 5)
        # # 1600 -> 512
        # x = F.leaky_relu(self.fc1(x))        
        # # don't relu the last layer
        # x = self.fc2(x)

        return x