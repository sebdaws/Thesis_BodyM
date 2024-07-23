import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
import torchvision.transforms as T
from load_data import SegmentationDataset
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  
import argparse
import torchmetrics as TM
from enum import IntEnum

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class TrimapClasses(IntEnum):
    PERSON = 1

def plt_images(image, mask, pred):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred.cpu().squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

# Custom Sigmoid Head for the model
class SigmoidDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(SigmoidDeepLabHead, self).__init__(
            DeepLabHead(in_channels, num_classes),
            nn.Sigmoid()
        )

def iou_calc(true_mask, predicted_mask):
    intersection = np.logical_and(true_mask, predicted_mask)
    union = np.logical_or(true_mask, predicted_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou

def iou_score(output, target):
    smooth = 1e-6
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    output = output > 0.5  # Applying threshold to logits
    target = target > 0.5
    
    intersection = (output & target).sum()
    union = (output | target).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded 

def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze()
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded 

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def load_model(model_path, device):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.aux_classifier = None
    model.classifier = SigmoidDeepLabHead(2048, 1)  # Assuming binary classification
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def test_model(model, dataloader, device, preview=False):
    criterion = nn.BCELoss()
    all_preds = []
    all_targets = []
    iou_scores = []
    min_iou = 1

    # pixel_metric = TM.classification.MulticlassAccuracy(2, average='micro')
    # pixel_metric = pixel_metric.to(device)
    # iou = TM.classification.MulticlassJaccardIndex(2, average='micro', ignore_index=TrimapClasses.BACKGROUND)
    # iou = iou.to(device)   

    print('Testing model...')
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            preds = outputs.data.cpu().numpy()
            targets = masks.data.cpu().numpy()
            all_preds.extend(preds.ravel())
            all_targets.extend(targets.ravel())

            for j in range(images.size(0)):
                iou_score_val = iou_numpy(preds[j].squeeze(), targets[j].squeeze())
                iou_scores.append(iou_score_val)
                if iou_score_val < min_iou:
                    min_iou = iou_score_val

            if preview:
                plt_images(images[1], masks[1], outputs[1])
                return
            
            if (i+1)%1 == 0:
                print(f'Batch {i+1}/{len(dataloader)}, Loss: {iou_score_val}')
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    f1 = f1_score(all_targets > 0, all_preds > 0.5)
    auroc = roc_auc_score(all_targets, all_preds)
    return f1, auroc, (np.mean(iou_scores), min_iou)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', required=False, type=bool, default=False, help='Select whether to plot preview.')
    parser.add_argument('--weights', required=True, type=str, default='./trained_models/test_resnet50.pt', help='Specify path to weights.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model_path = args.weights
    model = load_model(model_path, device)

    model.eval()

    # Setup the test dataset and dataloader
    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = SegmentationDataset(
            root_dir=os.path.join('data','segmentation_dataset'),
            transform=transform
        )

    # Split the dataset into train, validation, and test
    total_count = len(dataset)
    train_count = int(0.8 * total_count)
    valid_count = int(0.10 * total_count)
    test_count = total_count - train_count - valid_count

    _, _, test_dataset = random_split(dataset, [train_count, valid_count, test_count])

    percent = 0.1
    if percent < 1:
        subset_size = int(percent*len(test_dataset))
        indices = list(range(len(test_dataset)))
        random_indices = random.sample(indices, subset_size)
        test_dataset = Subset(test_dataset, random_indices)

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)
    
    # Evaluate the model
    f1_score, auroc_score, ious = test_model(model, test_loader, device, preview=args.preview)
    print(f'Test F1 Score: {f1_score:.4f}')
    print(f'Test AUROC Score: {auroc_score:.4f}')
    print(f'Test IoU Score: {ious[0]:.4f}')
    print(f'Lowest IoU Score: {ious[1]:.4f}')

if __name__ == '__main__':
    main()