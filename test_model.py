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
import torch.nn.functional as F
import pandas as pd

from utils import plt_images, iou_calc, pixel_accuracy
from get_model import load_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def test_model(model, dataloader, args, device, finetune=True, preview=False):
    criterion = nn.BCELoss()
    all_preds = []
    all_targets = []

    iou_scores = []
    min_iou = 1

    pixacc_scores = []
    min_acc = 1
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            if finetune==False:
                outputs = model(images)['out'].argmax(1).unsqueeze(1)
                outputs = (outputs == 15).float()
                preds = (outputs > 0.5).byte().cpu().numpy()
            else:
                if args.pointrend:
                    outputs = model(images)
                    outputs = F.sigmoid(outputs)
                else:
                    outputs = model(images)['out']
                preds = outputs.data.cpu().numpy()
            # print(outputs)
            targets = masks.data.cpu().numpy()
            loss = criterion(outputs, masks)
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
                print(loss, iou_score_val, pixacc_score_val)
                plt_images(images[0], masks[0], outputs[0])
                return
            
            # if (i+1)%10 == 0:
            #     print(f'Batch {i+1}/{len(dataloader)}, IoU: {iou_score_val:.4f}, Acc: {pixacc_score_val:.4f}')
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    f1 = f1_score(all_targets > 0, all_preds > 0.5)
    
    return f1, (np.mean(iou_scores), min_iou), (np.mean(pixacc_scores), min_acc)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str, default='resnet50', help='Specify path to weights.')
    parser.add_argument('--pointrend', action='store_true', default=False)
    parser.add_argument('--weights', required=False, type=str, default=False, help='Specify path to weights.')
    parser.add_argument('--preview', action='store_true', help='Select whether to plot preview.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model_path = args.weights
    model = load_model(args.backbone, finetune=args.weights, pointrend=args.pointrend)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    im_transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    dataset = SegmentationDataset(
            root_dir=os.path.join('data','segmentation_dataset'),
            im_transform=im_transform,
            mask_transform=mask_transform
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
    
    print(f'Testing model, {len(test_dataset)} samples')
    f1_score, ious, accs = test_model(model, test_loader, args, device, finetune=args.weights, preview=args.preview)
    print(f'Test F1 Score: {f1_score:.4f}')
    # print(f'Test AUROC Score: {auroc_score:.4f}')
    print(f'Test IoU Score: {ious[0]:.4f}')
    print(f'Test Pixel Accuracy: {accs[0]:.4f}\n')
    print(f'Lowest IoU Score: {ious[1]:.4f}')
    print(f'Lowest Pixel Accuracy: {accs[1]:.4f}')

    if model_path:
        model_name = os.path.basename(model_path)
    else:
        model_name = 'baseline'

    metrics = {
        'model_path': model_name,
        'backbone': args.backbone,
        'Iou': ious[0],
        'Pix_Acc': accs[0],
        'F1': f1_score,
        'min_IoU': ious[1],
        'min_Pix_Acc': accs[1]
    }

    metrics_df = pd.DataFrame([metrics])

    # Create directory if it doesn't exist
    os.makedirs('test_metrics', exist_ok=True)

    # Save to CSV
    csv_path = os.path.join('test_metrics', model_name)
    if not os.path.isfile(csv_path):
        metrics_df.to_csv(csv_path, index=False)
    else:
        metrics_df.to_csv(csv_path, mode='a', header=False, index=False)

    print(f'Metrics saved to {csv_path}')

if __name__ == '__main__':
    main()