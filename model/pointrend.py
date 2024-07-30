import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101

from model.sampling_points import sampling_points, point_sample

# Define the Bottleneck block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the ResNet Backbone
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = resnet101(pretrained=True)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3
        self.res5 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        res2 = self.res2(x)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        return res2, res3, res4, res5

# Define the FPN
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.fpn_lateral2 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.fpn_output2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_lateral3 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.fpn_output3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_lateral4 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.fpn_output4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_lateral5 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)
        self.fpn_output5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, res2, res3, res4, res5):
        lat5 = self.fpn_lateral5(res5)
        lat4 = self.fpn_lateral4(res4)
        lat3 = self.fpn_lateral3(res3)
        lat2 = self.fpn_lateral2(res2)

        fpn5 = self.fpn_output5(lat5)
        fpn4 = self.fpn_output4(lat4 + F.interpolate(lat5, size=lat4.shape[-2:], mode='nearest'))
        fpn3 = self.fpn_output3(lat3 + F.interpolate(fpn4, size=lat3.shape[-2:], mode='nearest'))
        fpn2 = self.fpn_output2(lat2 + F.interpolate(fpn3, size=lat2.shape[-2:], mode='nearest'))

        return fpn2, fpn3, fpn4, fpn5

# Define the Semantic FPN Head
class SemSegFPNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemSegFPNHead, self).__init__()
        self.p5 = self._make_layer(in_channels, 128)
        self.p4 = self._make_layer(in_channels, 128)
        self.p3 = self._make_layer(in_channels, 128)
        self.p2 = self._make_layer(in_channels, 128)
        self.predictor = nn.Conv2d(128, num_classes, kernel_size=1)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fpn2, fpn3, fpn4, fpn5):
        p5 = self.p5(fpn5)
        p4 = self.p4(fpn4) + F.interpolate(p5, size=fpn4.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.p3(fpn3) + F.interpolate(p4, size=fpn3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.p2(fpn2) + F.interpolate(p3, size=fpn2.shape[-2:], mode='bilinear', align_corners=False)
        return self.predictor(p2)

# Define the Point Head
class PointHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointHead, self).__init__()
        self.fc1 = nn.Conv1d(in_channels, 256, kernel_size=1)
        self.fc2 = nn.Conv1d(256, 256, kernel_size=1)
        self.fc3 = nn.Conv1d(256, 256, kernel_size=1)
        self.predictor = nn.Conv1d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.predictor(x)

# Define the complete PointRend model
class PointRendSemanticFPN(nn.Module):
    def __init__(self, num_classes):
        super(PointRendSemanticFPN, self).__init__()
        self.backbone = ResNetBackbone()
        self.fpn = FPN([256, 512, 1024, 2048], 256)
        self.sem_seg_head = SemSegFPNHead(256, num_classes)
        self.point_head = PointHead(256 + num_classes, num_classes)  # 256 (FPN output) + 19 (coarse seg head output)

    def forward(self, x):
        res2, res3, res4, res5 = self.backbone(x)
        fpn2, fpn3, fpn4, fpn5 = self.fpn(res2, res3, res4, res5)
        coarse = self.sem_seg_head(fpn2, fpn3, fpn4, fpn5)

        # Sample points
        num_points = 1000  # Define number of points to sample
        _, points = sampling_points(coarse, num_points)

        # Extract point features
        fine_grained_features = point_sample(fpn2, points)
        coarse_features = point_sample(coarse, points)

        point_features = torch.cat([fine_grained_features, coarse_features], dim=1)

        # Apply point head
        point_features = point_features.permute(0, 2, 1)  # (batch_size, feature_dim, num_points)
        refined = self.point_head(point_features)

        # Reshape refined to match the coarse output shape
        refined = refined.permute(0, 2, 1).contiguous()
        refined_output = F.grid_sample(coarse, 2.0 * points - 1.0, mode='bilinear', align_corners=False)
        refined_output = refined_output.view(coarse.shape[0], coarse.shape[1], -1)
        refined_output.scatter_(2, points.long(), refined)

        return refined_output

