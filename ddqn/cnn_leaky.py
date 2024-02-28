import torch.nn as nn
import torch.nn.functional as F

class CNNLeaky(nn.Module):
    def __init__(self, output_size):
        super(CNNLeaky, self).__init__()

        # 11 x 11 -> 10 x 10
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        # 10 x 10 -> 9 x 9
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(64)
        # 9 x 9 -> 8 x 8
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.batch_norm3 = nn.BatchNorm2d(128)
        # 8 x 8 -> 8 x 8
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        # 256 * 7 * 7 -> 2304
        self.fc1 = nn.Linear(16384, 512)

        self.fc2 = nn.Linear(512, output_size)

        # # 11 x 11 -> 9 x 9
        # self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3)
        # # 9 x 9 -> 7 x 7
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # # 7 x 7 -> 5 x 5
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # # 5 x 5 -> 3 x 3
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        # # 256 * 3 * 3 -> 2304
        # self.fc1 = nn.Linear(2304, 512)

        # self.fc2 = nn.Linear(512, output_size)

        # # (67-3)/2 + 1 -> 33 x 33
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        # # (33-3)/2 + 1 -> 16 x 16
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        # # (16-3)/1 + 1 -> 14 x 14
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # # (14-3)/1 + 1 -> 12 x 12
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        # # 256 * 12 * 12 -> 4608
        # self.fc1 = nn.Linear(36864, 512)

        # self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        if (len(x.shape) == 3):
            x = x.unsqueeze(0)

        # 11 x 11 -> 9 x 9
        x = F.leaky_relu(self.batch_norm1(self.conv1(x)))
        # 9 x 9 -> 7 x 7
        x = F.leaky_relu(self.batch_norm2(self.conv2(x)))
        # 7 x 7 -> 5 x 5
        x = F.leaky_relu(self.batch_norm3(self.conv3(x)))
        # 5 x 5 -> 3 x 3
        x = F.leaky_relu(self.batch_norm4(self.conv4(x)))
        # # 256 * 3 * 3 -> 2304
        # x = x.view(-1, 2304)
        # 256 * 11 * 11 -> 30976
        x = x.view(-1, 16384)

        x = F.leaky_relu(self.fc1(x))
        # don't relu the last layer
        x = self.fc2(x)

        return x