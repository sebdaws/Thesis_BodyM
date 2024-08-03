import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# Define the Body Measurement Estimation Network
class MeasureNet(nn.Module):
    def __init__(self, num_outputs=14, in_channels=3):
        super(MeasureNet, self).__init__()
        
        # MNASNet backbone
        self.backbone = models.mnasnet1_0(weights='DEFAULT')
        self.backbone.classifier = nn.Identity()
        self.backbone.layers[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # MLP for final regression
        self.mlp = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        # x = torch.cat([x, m], dim=1)
        # print(x.shape)
        x = self.mlp(x)
        return x