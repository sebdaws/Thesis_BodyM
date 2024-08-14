import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os

from get_model import load_model
from utils import plt_images

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

def save_prediction(output, model_name, output_dir='preview'):
    """Save the model prediction to a file."""
    output_image = output.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the output image with the model's base name
    output_path = os.path.join(output_dir, f"{model_name}.png")
    Image.fromarray(output_image).save(output_path)
    print(f"Prediction saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str, default='resnet50', help='Specify path to weights.')
    parser.add_argument('--model_path', required=False, type=str, default=False, help='Specify path to weights.')
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
            output = (output == 15).float()
            pred = (output > 0.5).byte().cpu().numpy()
        else:
            output = model(image)['out']
            # pred = output.data.cpu().numpy()
            pred = (output > 0.5).float()
        
        # Save the output prediction
        save_prediction(pred, model_base_name)

    # Plot the results
    plt_images(image.squeeze(0), mask, output)

if __name__ == '__main__':
    main()