import torch.nn as nn
import torch.nn.functional as F

class CNNLeakySameV2(nn.Module):
    def __init__(self, output_size):
        super(CNNLeakySameV2, self).__init__()

        # 11 x 11 -> 11 x 11
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=5, padding=2)
        # 11 x 11 -> 9 x 9
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # 9 x 9 -> 7 x 7
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        # 7 x 7 -> 5 x 5
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        # 5 x 5 * 512 -> 12,800 -> 512
        self.fc1 = nn.Linear(12800, 512)        
        # 512 -> output_size
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x.view(-1, 12800)
        
        x = F.leaky_relu(self.fc1(x))

        x = self.fc2(x)

        return x