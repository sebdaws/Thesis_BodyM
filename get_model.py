import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights, DeepLabHead
from model.pointrend import PointRendSemanticFPN
# from model.sampling_points import sampling_points, point_sample

class SigmoidDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(SigmoidDeepLabHead, self).__init__(
            DeepLabHead(in_channels, num_classes),
            nn.Sigmoid()
        )

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

class PointRendDeepLabV3(nn.Module):
    def __init__(self, backbone='resnet101', num_classes=1):
        super(PointRendDeepLabV3, self).__init__()
        if backbone=='resnet101':
            self.deeplabv3 = deeplabv3_resnet101(weight='DEFAULT')
            self.backbone = self.deeplabv3.backbone
            self.classifier = self.deeplabv3.classifier

        self.point_head = PointHead(2048 + num_classes, num_classes)  # 256 (FPN output) + 19 (coarse seg head output)

    def forward(self, x):
        features = self.backbone(x)
        coarse = self.classifier(features['out'])

        # Sample points
        num_points = 1000
        _, points = sampling_points(coarse, num_points)

        # Extract point features
        fine_grained_features = point_sample(features['out'], points.unsqueeze(0).repeat(features['out'].size(0), 1, 1))
        coarse_features = point_sample(coarse, points.unsqueeze(0).repeat(coarse.size(0), 1, 1))

        print(fine_grained_features.shape)
        print(coarse_features.shape)

        point_features = torch.cat([fine_grained_features, coarse_features], dim=1)

        # Apply point head
        print(point_features.shape)
        # point_features = point_features.permute(0, 2, 1)
        # print(point_features.shape)
        refined = self.point_head(point_features)

        # Reshape refined to match the coarse output shape
        refined = refined.permute(0, 2, 1).contiguous()
        refined_output = coarse.clone()
        refined_output = F.interpolate(refined_output, size=refined_output.shape[-2:], mode='bilinear', align_corners=False)
        print(points.shape)
        print(refined.shape)
        refined_output.scatter_(2, points.long(), refined)

        return refined_output

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, C, H, W = mask.shape

    if C < 2:
        mask = mask.repeat(1, 2, 1, 1)

    mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)


def load_model(backbone, freeze=False, pointrend=False):
    if pointrend:
        model = PointRendSemanticFPN(num_classes=1)
        print(f'Semantic FPN with PointRend')
    else:
        if backbone == 'resnet50':
            model_weights = DeepLabV3_ResNet50_Weights.DEFAULT
            model = deeplabv3_resnet50(weights=model_weights)
            channels = 2048
        elif backbone == 'resnet101':
            model_weights = DeepLabV3_ResNet101_Weights.DEFAULT
            model = deeplabv3_resnet101(weights=model_weights)
            channels = 2048
        elif backbone == 'mobilenet':
            model_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            model = deeplabv3_mobilenet_v3_large(weights=model_weights)
            channels = 960
        else:
            raise NameError('Chosen weights not available')
    
        print(f'DeepLabV3, {backbone} backbone')

        if freeze:
            model.aux_classifier = None
            model.classifier = SigmoidDeepLabHead(channels, 1)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True

    return model