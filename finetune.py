import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights, DeepLabHead
import time
import os
import copy
import tqdm
import csv
from sklearn.metrics import f1_score, roc_auc_score
from load_data import SegmentationDataset
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SigmoidDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(SigmoidDeepLabHead, self).__init__(
            DeepLabHead(in_channels, num_classes),
            nn.Sigmoid()
        )

def train_model(model, dataloader, optimizer, bpath, num_epochs, device):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    model.to(device)

    # params_to_update = [param for param in model.parameters() if param.requires_grad]
    # optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        model.train()  # Set model to training mode
        phase = 'Train'
        # Iterate over data.
        batch_loss = 0
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(phase == 'Train'):
                outputs = model(images)['out']
                # print(outputs.shape, masks.shape)
                loss = criterion(outputs, masks)
                y_pred = outputs.data.cpu().numpy().ravel()
                y_true = masks.data.cpu().numpy().ravel()
                for name, metric in metrics.items():
                    if name == 'f1_score':
                        # Use a classification threshold of 0.1
                        batchsummary[f'{phase}_{name}'].append(
                            metric(y_true > 0, y_pred > 0.1))
                    else:
                        batchsummary[f'{phase}_{name}'].append(
                            metric(y_true.astype('uint8'), y_pred))

                # backward + optimize only if in training phase
                loss.backward()
                batch_loss += loss
                optimizer.step()

                if (i+1)%10 == 0:
                    print(f'Batch {i+1}/{len(dataloader)}, Loss: {batch_loss/10}')
                    batch_loss = 0

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
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    weights = 'resnet50'

    if weights == 'resnet50':
        model_weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=model_weights).to(device)
    elif weights == 'resnet101':
        model_weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=model_weights).to(device)
    elif weights == 'mobilenet':
        model_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=model_weights).to(device)
    else:
        raise NameError('Chosen weights not available')
    
    print(f'DeepLabV3, {weights} backbone')

    model.aux_classifier = None

    # Assuming you're using a ResNet50 backbone as in your example:
    num_classes = 1  # Since you're likely doing binary segmentation
    model.classifier = SigmoidDeepLabHead(2048, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    percent = 0.1

    transform = T.Compose([
            # transforms.Resize((256, 256)),
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

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_count, valid_count, test_count])

    if percent < 1:
        subset_size = int(percent*len(train_dataset))
        indices = list(range(len(train_dataset)))
        random_indices = random.sample(indices, subset_size)
        train_dataset = Subset(train_dataset, random_indices)

    # Create data loaders for each split
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

    trained_model = train_model(model, train_loader, optimizer, Path('./test'), 1, device)

    torch.save(trained_model.state_dict(), os.path.join('trained_models', 'test_resnet50.pt'))

if __name__ == '__main__':
    main()