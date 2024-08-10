import numpy as np
import torch
import torch.nn as nn
import time
import os
import tqdm
import csv
from sklearn.metrics import f1_score
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path
import argparse

from load_data import SegmentationDataset
from utils import iou_calc, pixel_accuracy
from get_model import load_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_metrics(preds, targets, metrics, batchsummary, phase, batch_size):
    for name, metric in metrics.items():
        if name == 'f1_score':
            # Use a classification threshold of 0.1
            f1 = metric(targets.ravel() > 0, preds.ravel() > 0.1)
            batchsummary[f'{phase}_{name}'].append(f1)
        else:
            score_ls = []
            for j in range(batch_size):
                score = metric(targets[j], preds[j])
                score_ls.append(score)
            batchsummary[f'{phase}_{name}'].append(np.mean(score_ls))
    
    return batchsummary

def train_model(model, trainloader, validloader, optimizer, metpath, num_epochs, device):
    since = time.time()
    best_loss = 1e10
    model.to(device)
    criterion = nn.BCELoss()

    metrics = {'f1_score': f1_score, 'pix_acc': pixel_accuracy, 'IoU': iou_calc}
    
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Valid_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Valid_{m}' for m in metrics.keys()]
    
    with open(metpath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        batchsummary = {a: [0] for a in fieldnames}

        #--------Train--------#
        model.train()
        phase = 'Train'
        pbar = tqdm.tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch')

        batch_loss = 0
        for i, (images, masks) in enumerate(trainloader):
            images = images.to(device)
            masks = masks.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)['out']

            loss = criterion(outputs, masks)
            preds = outputs.data.cpu().numpy()
            targets = masks.data.cpu().numpy()

            loss.backward()
            batch_loss += loss
            optimizer.step()

            batchsummary = train_metrics(preds, targets, metrics, batchsummary, phase, images.size(0))

            pbar.set_postfix({'Loss': f'{batch_loss/(i+1):.4f}'})
            pbar.update()
        
        batchsummary[f'{phase}_loss'] = loss.item()

        pbar.close()

        #--------Validation--------#
        phase = 'Valid'
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in validloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)['out']
                preds = outputs.data.cpu().numpy()
                targets = masks.data.cpu().numpy()

                loss = criterion(outputs, masks)

                running_loss += loss.item()

                batchsummary = train_metrics(preds, targets, metrics, batchsummary, phase, images.size(0))

                pbar.set_postfix({'epoch': epoch, 'phase': 'valid', 'loss': f'{running_loss / len(validloader):.4f}'})
                pbar.update()

        pbar.close()

        valid_loss = running_loss / len(validloader)
        if valid_loss < best_loss:
            best_loss = valid_loss
        
        batchsummary['epoch'] = epoch
        batchsummary[f'{phase}_loss'] = valid_loss
        print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(metpath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', required=False, type=float, default=1.0, help='Select the percentage of the training set to train on.')
    parser.add_argument('--backbone', required=False, choices=['resnet50', 'resnet101', 'mobilenet'], default='resnet50', help='Specify wether to train the model or test it.')
    parser.add_argument('--num_epochs', required=False, type=int, default=3, help='Number of epochs on which to train. (default is 20)')
    parser.add_argument('--batch_size', required=False, type=int, default=4, help='Size of batch for training. (default is 4)')
    parser.add_argument('--resize', required=False, type=int, default=512, help='Size of images to resize to.')
    parser.add_argument('--freeze', required=False, action='store_true', default=False, help='Specify whether to freeze backbone weights.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Device: {device}')


    model = load_model(args.backbone, finetune=True, freeze=args.freeze).to(device)

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params_to_update, lr=0.001)#, momentum=0.9)


    im_transform = T.Compose([
            T.Resize((args.resize, args.resize)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    mask_transform = T.Compose([
        T.Resize((args.resize, args.resize)),
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

    train_dataset, valid_dataset, _ = random_split(dataset, [train_count, valid_count, test_count])

    percent = args.percent

    if percent < 1:
        subset_size = int(percent*len(train_dataset))
        indices = list(range(len(train_dataset)))
        random_indices = random.sample(indices, subset_size)
        train_dataset = Subset(train_dataset, random_indices)

    # Create data loaders for each split
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.freeze:
        str_fr = 'f'
    else:
        str_fr = 'nf'

    metrics_folder = Path('./finetune_metrics')
    if not metrics_folder.exists():
        metrics_folder.mkdir()
    metrics_file = os.path.join(metrics_folder, f'{args.backbone}_{str_fr}_{int(args.percent*100)}p_{args.num_epochs}e_{args.resize}px.csv')

    trained_model = train_model(
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        metrics_file, 
        args.num_epochs, 
        device
        )

    save_folder = Path('./trained_models')
    if not save_folder.exists():
        save_folder.mkdir()

    torch.save(trained_model.state_dict(), os.path.join(save_folder, f'{args.backbone}_{str_fr}_{int(args.percent*100)}p_{args.num_epochs}e_{args.resize}px.pt'))

if __name__ == '__main__':
    main()