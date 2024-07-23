import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score
import torchvision.transforms as T
from load_data import SegmentationDataset
import random
import argparse

from utils import plt_images
from model import load_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def iou_calc(true_mask, predicted_mask, threshold=0.5):
    predicted_mask = (predicted_mask > threshold).astype(float)
    intersection = np.logical_and(true_mask, predicted_mask)
    union = np.logical_or(true_mask, predicted_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou

def pixel_accuracy(true_mask, predicted_mask, threshold=0.5):
    """
    Calculate pixel accuracy by comparing the predicted mask (after thresholding)
    with the ground truth mask.
    
    Parameters:
    true_mask (numpy.ndarray): The ground truth binary mask.
    predicted_mask (numpy.ndarray): The predicted probabilities.
    threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
    float: The pixel accuracy score.
    """
    predicted_mask = (predicted_mask > threshold).astype(int)
    correct_predictions = (predicted_mask == true_mask).sum()
    total_pixels = true_mask.size
    pixel_acc = correct_predictions / total_pixels
    return pixel_acc

def test_model(model, dataloader, device, preview=False):
    criterion = nn.BCELoss()
    all_preds = []
    all_targets = []

    iou_scores = []
    min_iou = 1

    pixacc_scores = []
    min_acc = 1

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
                iou_score_val = iou_calc(targets[j], preds[j])
                iou_scores.append(iou_score_val)
                if iou_score_val < min_iou:
                    min_iou = iou_score_val
                
                pixacc_score_val = pixel_accuracy(targets[j], preds[j])
                pixacc_scores.append(pixacc_score_val)
                if pixacc_score_val < min_acc:
                    min_acc = pixacc_score_val

            if preview:
                plt_images(images[1], masks[1], outputs[1])
                return
            
            if (i+1)%10 == 0:
                print(f'Batch {i+1}/{len(dataloader)}, IoU: {iou_score_val:.4f}, Acc: {pixacc_score_val:.4f}')
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    f1 = f1_score(all_targets > 0, all_preds > 0.5)
    auroc = roc_auc_score(all_targets, all_preds)
    return f1, auroc, (np.mean(iou_scores), min_iou), (np.mean(pixacc_scores), min_acc)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', required=False, type=bool, default=False, help='Select whether to plot preview.')
    parser.add_argument('--weights', required=False, type=str, default=False, help='Specify path to weights.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model_path = args.weights
    model = load_model('resnet50')
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Setup the test dataset and dataloader
    transform = T.Compose([
        T.Resize((512, 512)),
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

    percent = 1
    if percent < 1:
        subset_size = int(percent*len(test_dataset))
        indices = list(range(len(test_dataset)))
        random_indices = random.sample(indices, subset_size)
        test_dataset = Subset(test_dataset, random_indices)

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)
    
    # Evaluate the model
    f1_score, auroc_score, ious, accs = test_model(model, test_loader, device, preview=args.preview)
    print(f'Test F1 Score: {f1_score:.4f}')
    print(f'Test AUROC Score: {auroc_score:.4f}')
    print(f'Test IoU Score: {ious[0]:.4f}')
    print(f'Lowest IoU Score: {ious[1]:.4f}')
    print(f'Test Pixel Accuracy: {accs[0]:.4f}')
    print(f'Lowest Pixel Accuracy: {accs[1]:.4f}')

if __name__ == '__main__':
    main()