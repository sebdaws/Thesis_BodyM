import torch
import torchvision
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights
from torchvision import transforms
import torchvision.transforms as T
from load_data import SegmentationDataset
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import random
import os
import argparse

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def plt_images(image, true_mask, predicted_mask):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.show()

def train_model(model, device, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, true_masks in train_loader:
            images = images.to(device)
            true_masks = true_masks.to(device).long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = torch.nn.functional.cross_entropy(outputs, true_masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    print("Training completed.")

def evaluate_model(data_loader, model, device, preview=False):
    model.eval()
    model.to(device)
    iou_scores = []
    min_iou = 1
    
    for images, true_masks in data_loader:
        images = images.to(device)
        true_masks = true_masks.squeeze(1).numpy()  # Squeezing batch dimension for masks

        with torch.no_grad():
            outputs = model(images)['out'].argmax(1).byte().cpu().numpy()
        
        for idx in range(images.size(0)):
            true_mask = true_masks[idx]
            predicted_mask = (outputs[idx] == 15).astype(np.uint8)
        
            intersection = np.logical_and(true_mask, predicted_mask)
            union = np.logical_or(true_mask, predicted_mask)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
            iou_scores.append(iou)
            if iou < min_iou:
                min_iou = iou
            
            if preview:
                plt_images(images[idx], true_mask, predicted_mask)
                return

        print(f"Processed {len(iou_scores)}/{len(data_loader.dataset)} - Current IoU: {iou:.4f}")
    
    # Calculate the mean IoU across all samples
    mean_iou = np.mean(iou_scores)
    return [mean_iou, min_iou]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'test'], help='Specify wether to train the model or test it.')
    parser.add_argument('--percent', required=False, type=float, default=1.0, help='Select the percentage of the training set to train on.')
    parser.add_argument('--model', type=str, default='deeplabv3', help='Choose model to test.')
    parser.add_argument('--weights', required=False, choices=['resnet50', 'resnet101', 'mobilenet'], default='resnet50', help='Specify wether to train the model or test it.')
    parser.add_argument('--num_epochs', required=False, type=int, default=20, help='Number of epochs on which to train. (default is 20)')
    args = parser.parse_args()

    task_txt = f'Task: {args.mode} {args.model}'
    if args.mode == 'train':
        task_txt += f', {int(100*args.percent)}% of training set, {args.num_epochs} epochs'
    print(task_txt)
    
    # Define any transformations you want to apply to both images and masks
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Initialising dataset...')
    # Initialize the dataset
    dataset = SegmentationDataset(
        root_dir=os.path.join('CAESAR','output_dataset'),
        transform=transform
    )
    
    # Split the dataset into train, validation, and test
    total_count = len(dataset)
    train_count = int(0.8 * total_count)
    valid_count = int(0.10 * total_count)
    test_count = total_count - train_count - valid_count

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_count, valid_count, test_count])
    
    if args.percent < 1:
        subset_size = int(args.percent*len(train_dataset))
        indices = list(range(len(train_dataset)))
        random_indices = random.sample(indices, subset_size)
        train_dataset = Subset(train_dataset, random_indices)

    # Create data loaders for each split
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print('Data ready.')

    # data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example of setting up the model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'Device: {device}')
    
    print('Loading model...')
    
    if args.weights == 'resnet50':
        model_weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=model_weights).to(device)
    elif args.weights == 'resnet101':
        model_weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=model_weights).to(device)
    elif args.weights == 'mobilenet':
        model_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=model_weights).to(device)
    else:
        raise NameError('Chosen weights not available')
    
    print('Model loaded')
    
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, device, train_loader, optimizer, num_epochs=args.num_epochs)
        if not os.path.exists('trained_models'):
            os.mkdir('trained_models')
        save_path = os.path.join('trained_models', f'{args.model}_{int(100*args.percent)}')
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    elif args.mode == 'test':
        print(f'Testing {args.model} with {args.weights}, {test_count} test samples, on {device}')

        # Assuming 'data_loader' is an iterable over preprocessed image and mask pairs
        iou_scores = evaluate_model(test_loader, model, device, preview=False)
        print(f"Mean IoU: {iou_scores[0]:.4f}")
        print(f"Lowest IoU: {iou_scores[1]:.4f}")
        
    else:
        raise AttributeError("Incorrect mode, should be 'train'or 'test'")

if __name__ == '__main__':
    main()