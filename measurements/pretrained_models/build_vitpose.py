import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrained_models.vit import ViT, PatchEmbed
from pretrained_models.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

models = {
    "ViTPose_base_coco_256x192": dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict()),
    
"ViTPose_base_simple_coco_256x192":  dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=0,
        num_deconv_filters=[],
        num_deconv_kernels=[],
        upsample=4,
        extra=dict(final_conv_kernel=3, ),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate_kernel=11,
        use_udp=True))

    }

def modify_patch_embed(patch_embed, new_in_channels):
    # Create a new convolution layer with the same properties but different in_channels
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=patch_embed.proj.out_channels,
        kernel_size=patch_embed.proj.kernel_size,
        stride=patch_embed.proj.stride,
        padding=patch_embed.proj.padding,
        bias=patch_embed.proj.bias is not None
    )

    # Copy the existing weights to the new conv layer
    with torch.no_grad():
        if new_in_channels > patch_embed.proj.in_channels:
            new_conv.weight[:, :patch_embed.proj.in_channels, :, :] = patch_embed.proj.weight
            if new_in_channels > patch_embed.proj.in_channels:
                # Initialize the additional channels' weights as needed
                new_conv.weight[:, patch_embed.proj.in_channels:, :, :] = torch.mean(patch_embed.proj.weight, dim=1, keepdim=True)
        else:
            new_conv.weight = patch_embed.proj.weight[:, :new_in_channels, :, :]

    return new_conv



class ExtendedPatchEmbed(nn.Module):
    def __init__(self, original_patch_embed, additional_in_channels):
        super(ExtendedPatchEmbed, self).__init__()
        self.original_patch_embed = original_patch_embed
        self.additional_conv = nn.Conv2d(
            in_channels=additional_in_channels,
            out_channels=original_patch_embed.proj.out_channels,
            kernel_size=original_patch_embed.proj.kernel_size,
            stride=original_patch_embed.proj.stride,
            padding=original_patch_embed.proj.padding,
            bias=original_patch_embed.proj.bias is not None
        )

    def forward(self, x1, x2):
        # Split the input into the original and additional channels
        x_original = x[:, :3, :, :]
        x_additional = x[:, 3:, :, :]

        # Process the original channels and additional channels separately
        x1, (Hp, Wp) = self.original_patch_embed(x_original)  # The original patch embedding returns (tensor, (Hp, Wp))
        x2 = self.additional_conv(x_additional)

        # Ensure the spatial dimensions match before summing
        if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
            x2 = F.interpolate(x2, size=(Hp, Wp), mode='bilinear', align_corners=False)

        # Combine the outputs by summing them
        x = x1 + x2

        # Return the combined tensor and the original (Hp, Wp) tuple
        return x

def build_vitpose(model_name,checkpoint=None):
    try:
        model = models[model_name]
    except:
        raise ValueError('not a correct config')

        
    head = TopdownHeatmapSimpleHead(in_channels=model['keypoint_head']['in_channels'], 
                                    out_channels=model['keypoint_head']['out_channels'],
                                    num_deconv_filters=model['keypoint_head']['num_deconv_filters'],
                                    num_deconv_kernels=model['keypoint_head']['num_deconv_kernels'],
                                    num_deconv_layers=model['keypoint_head']['num_deconv_layers'],
                                    extra=model['keypoint_head']['extra'])

    backbone = ViT(img_size=model['backbone']['img_size'],
                patch_size=model['backbone']['patch_size']
                ,embed_dim=model['backbone']['embed_dim'],
                depth=model['backbone']['depth'],
                num_heads=model['backbone']['num_heads'],
                ratio = model['backbone']['ratio'],
                mlp_ratio=model['backbone']['mlp_ratio'],
                qkv_bias=model['backbone']['qkv_bias'],
                drop_path_rate=model['backbone']['drop_path_rate']
                )
    
    class VitPoseModel(nn.Module):
        def __init__(self,backbone,keypoint_head):
            super(VitPoseModel, self).__init__()
            self.backbone = backbone
            self.keypoint_head = keypoint_head
        def forward(self,x):
            x = self.backbone(x)
            x = self.keypoint_head(x)
            return x
    
    pose = VitPoseModel(backbone, head)
    if checkpoint is not None:
        check = torch.load(checkpoint, )
        pose.load_state_dict(check['state_dict'], strict=False)
    return pose

def modify_model_for_additional_channels(model, num_additional_channels):
    # Modify the patch embedding layer after loading weights
    model.backbone.patch_embed = PatchEmbed(img_size=(256, 192), patch_size=16, in_chans=6, embed_dim=768, ratio=1)
    return model