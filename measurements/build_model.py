import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import timm

from pretrained_models.build_vitpose import build_vitpose

class MeasureNet(nn.Module):
    def __init__(self, num_outputs=14, num_m=1, m_inputs=True):
        super(MeasureNet, self).__init__()
        mlp_inputs = 1280
        # MNASNet backbone
        self.backbone = models.mnasnet1_0(weights='DEFAULT')
        self.backbone.classifier = nn.Identity()
        if m_inputs:
            self.backbone.layers[0] = nn.Conv2d(3+num_m, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            mlp_inputs += num_m
        # MLP for final regression
        self.mlp = nn.Sequential(
            nn.Linear(mlp_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x, m=None):
        x = self.backbone(x)
        if m is not None:
            m = m.to(x.dtype)
            x = torch.cat([x, m], dim=1)
        x = self.mlp(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_features, num_outputs=15):
        super(MLPHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.fc(x)

class MeasureViT(nn.Module):
    def __init__(self, vitpose_name, vitpose_path=None, num_outputs=15, num_m=1):
        super(MeasureViT, self).__init__()
        original_model = build_vitpose(vitpose_name, vitpose_path)
        self.backbone = original_model.backbone
        in_features = 768 * 16 * 12 + num_m
        self.mlp_head = MLPHead(in_features=in_features, num_outputs=num_outputs)

    def forward(self, x, m=None):
        x = self.backbone(x)
        if m is not None:
            m = m.to(x.dtype)
            x = torch.cat([x, m], dim=1)
        x = self.mlp_head(x)
        return x
