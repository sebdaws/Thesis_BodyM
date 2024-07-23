import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights, DeepLabHead

class SigmoidDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(SigmoidDeepLabHead, self).__init__(
            DeepLabHead(in_channels, num_classes),
            nn.Sigmoid()
        )

def load_model(net):

    if net == 'resnet50':
        model_weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=model_weights)
    elif net == 'resnet101':
        model_weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=model_weights)
    elif net == 'mobilenet':
        model_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=model_weights)
    else:
        raise NameError('Chosen weights not available')
    
    print(f'DeepLabV3, {net} backbone')

    model.aux_classifier = None
    model.classifier = SigmoidDeepLabHead(2048, 1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {size_all_mb:.3f}MB, {total_trainable_params:,} training parameters.")

    return model