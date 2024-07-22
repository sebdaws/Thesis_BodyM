import numpy as np
import torch
import torch.nn as nn
import time
import os
import tqdm
import csv
from sklearn.metrics import f1_score, roc_auc_score
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path
import argparse

from load_data import SegmentationDataset
from model import load_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_model(model, trainloader, validloader, optimizer, bpath, num_epochs, device):
    since = time.time()
    best_loss = 1e10
    model.to(device)
    criterion = nn.BCELoss()

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    
    if not bpath.exists():
        bpath.mkdir()
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        # print('Epoch {}/{}'.format(epoch, num_epochs))
        # print('-' * 10)
        batchsummary = {a: [0] for a in fieldnames}

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
            # print(outputs.shape, masks.shape)
            loss = criterion(outputs, masks)
            y_pred = outputs.data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()
            loss.backward()
            batch_loss += loss
            optimizer.step()

            for name, metric in metrics.items():
                if name == 'f1_score':
                    # Use a classification threshold of 0.1
                    batchsummary[f'{phase}_{name}'].append(metric(y_true > 0, y_pred > 0.1))
                else:
                    batchsummary[f'{phase}_{name}'].append(metric(y_true.astype('uint8'), y_pred))

            pbar.set_postfix({'Loss': f'{batch_loss/(i+1):.4f}'})
            pbar.update()

            # if (i+1)%10 == 0:
            #     print(f'Batch {i+1}/{len(dataloader)}, Loss: {batch_loss/10}')
            #     batch_loss = 0

            # Validation Phase
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for images, masks in validloader:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)

                    running_loss += loss.item()
                    pbar.set_postfix({'epoch': epoch, 'phase': 'valid', 'loss': f'{running_loss / len(validloader):.4f}'})
                    pbar.update()

            valid_loss = running_loss / len(validloader)
            if valid_loss < best_loss:
                best_loss = valid_loss
        
        pbar.close()

        batchsummary['epoch'] = epoch
        epoch_loss = loss
        batchsummary[f'{phase}_loss'] = epoch_loss.item()
        print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', required=False, type=float, default=1.0, help='Select the percentage of the training set to train on.')
    parser.add_argument('--weights', required=False, choices=['resnet50', 'resnet101', 'mobilenet'], default='resnet50', help='Specify wether to train the model or test it.')
    parser.add_argument('--num_epochs', required=False, type=int, default=2, help='Number of epochs on which to train. (default is 20)')
    parser.add_argument('--batch_size', required=False, type=int, default=4, help='Size of batch for training. (default is 4)')
    parser.add_argument('--resize', required=False, type=int, default=256, help='Size of images to resize to.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Device: {device}')


    model = load_model(args.weights).to(device)

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    transform = T.Compose([
            T.Resize((args.resize, args.resize)),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    train_dataset, _, _ = random_split(dataset, [train_count, valid_count, test_count])

    percent = args.percent

    if percent < 1:
        subset_size = int(percent*len(train_dataset))
        indices = list(range(len(train_dataset)))
        random_indices = random.sample(indices, subset_size)
        train_dataset = Subset(train_dataset, random_indices)

    # Create data loaders for each split
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    trained_model = train_model(model, train_loader, optimizer, Path('./test'), args.num_epochs, device)

    torch.save(trained_model.state_dict(), os.path.join('trained_models', f'{args.weights}_{int(args.percent*100)}p_{args.num_epochs}e_{args.resize}px.pt'))

if __name__ == '__main__':
    main()