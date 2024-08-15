import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
from scipy.ndimage import gaussian_filter

from get_model import load_model
from utils import plt_images, iou_calc

def load_image(image_path):
    """Load an image from a file path and apply necessary transformations."""
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def load_mask(mask_path):
    """Load a mask from a file path and apply necessary transformations."""
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    mask = Image.open(mask_path).convert("L")
    mask = transform(mask).unsqueeze(0)
    return mask

def overlay_mask_on_image(image, mask, iou_score=None, color=[0, 255, 0], alpha=0.5):
    """Overlay the mask on the image and add IoU score as text."""
    # Denormalize the image
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0
    image_np = image_np.astype(np.uint8)

    mask_np = mask.squeeze().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8)       # Convert mask to binary

    overlay = image_np.copy()
    overlay[mask_np > 0] = overlay[mask_np > 0] * (1 - alpha) + np.array(color) * alpha

    overlay_image = Image.fromarray(overlay)

    if iou_score is not None:
        draw = ImageDraw.Draw(overlay_image)
        font = ImageFont.load_default(size=30)
        text = f"IoU: {iou_score:.4f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_position = (10, 10)
        draw.rectangle([text_position, (text_position[0] + text_bbox[2], text_position[1] + text_bbox[3])], fill="black")
        draw.text(text_position, text, fill="white", font=font)

    return overlay_image

def save_prediction(image, mask, model_name, iou_score, output_dir='preview'):
    """Save the model prediction to a file with mask overlay."""
    overlay = overlay_mask_on_image(image.squeeze(0), mask, iou_score)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the output image with the model's base name
    output_path = os.path.join(output_dir, f"{model_name}.png")
    overlay.save(output_path)
    print(f"Prediction with mask overlay saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str, default='resnet50', help='Specify path to weights.')
    parser.add_argument('--model_path', required=False, type=str, default=False, help='Specify path to weights.')
    parser.add_argument('--plot', default=False, action='store_true', help='Specify whether to plot the results.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model = load_model(args.backbone, finetune=args.model_path)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Get the base name of the model file (without extension)
    if args.model_path:
        model_base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    else:
        model_base_name = 'baseline_' + args.backbone

    # Configuration
    image_path = "data/segmentation_dataset/subfolder_1/images/csr0193a_front_view.png" 
    mask_path = "data/segmentation_dataset/subfolder_1/masks/csr0193a_front_view.png" 

    # Load the specific image
    image = load_image(image_path)
    image = image.to(device)

    mask = load_mask(mask_path)
    mask = mask.to(device)

    with torch.no_grad():
        if args.model_path==False:
            output = model(image)['out'].argmax(1).unsqueeze(1)
            output = (output == 15)
            pred = (output > 0.5).float()
        else:
            output = model(image)['out']
            # pred = output.data.cpu().numpy()
            pred = (output > 0.5).float()

        pred_np = pred.squeeze().cpu().numpy()

        # Apply Gaussian smoothing
        smoothed_pred_np = gaussian_filter(pred_np, sigma=1)  # Adjust sigma as needed

        # Convert the smoothed prediction back to a tensor
        smoothed_pred = torch.tensor(smoothed_pred_np).unsqueeze(0).unsqueeze(0).to(device)
        iou_score = iou_calc(mask.data.cpu().numpy(), smoothed_pred.cpu().numpy())

        # iou_score = iou_calc(mask.data.cpu().numpy(), pred.cpu().numpy())
        print(iou_score)
        
        save_prediction(image, pred, model_base_name, iou_score)

    if args.plot:
        plt_images(image.squeeze(0), mask, output)

if __name__ == '__main__':
    main()