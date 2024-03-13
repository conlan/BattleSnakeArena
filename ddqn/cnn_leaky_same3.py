import torch.nn as nn
import torch.nn.functional as F

class CNNLeakySameV3(nn.Module):
    def __init__(self, output_size):
        super(CNNLeakySameV3, self).__init__()

        # 11 x 11 -> 11 x 11
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, padding=1)
        # 11 x 11 -> 10 x 10
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        # 10 x 10 -> 9 x 9
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        # 9 x 9 - > 9 x 9
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        # 9 x 9 x 256 -> 20,736
        self.fc1 = nn.Linear(20736, 1024)
        # 1024 -> 512
        self.fc2 = nn.Linear(1024, 512)
        # 512 -> output_size
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x.view(-1, 20736)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = self.fc3(x)

        return x