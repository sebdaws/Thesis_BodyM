import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights, DeepLabHead

class SigmoidDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(SigmoidDeepLabHead, self).__init__(
            DeepLabHead(in_channels, num_classes),
            nn.Sigmoid()
        )

def load_model(net, finetune=False):

    if net == 'resnet50':
        model_weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=model_weights)
        channels = 2048
    elif net == 'resnet101':
        model_weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=model_weights)
        channels = 2048
    elif net == 'mobilenet':
        model_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=model_weights)
        channels = 960
    else:
        raise NameError('Chosen weights not available')
    
    print(f'DeepLabV3, {net} backbone')

    if finetune:
        model.aux_classifier = None
        model.classifier = SigmoidDeepLabHead(channels, 1)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    return model
