import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import explained_variance_score
import pandas as pd

from dataload import BodyMeasurementDataset
from build_model import MeasureNet

torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def check_for_nans(i, tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name} {i+1}")
        return True
    return False

# Function to calculate additional metrics
def r2_score(outputs, targets):
    ss_res = torch.sum((targets - outputs) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def calculate_metrics(outputs, targets):
    mse_loss = criterion(outputs, targets).item()
    rmse_loss = torch.sqrt(torch.tensor(mse_loss)).item()
    mae_loss = nn.L1Loss()(outputs, targets).item()
    explained_variance = explained_variance_score(targets.cpu().numpy(), outputs.cpu().detach().numpy())
    r2 = r2_score(outputs, targets)
    return rmse_loss, mae_loss, explained_variance, r2

def quantile_metrics(outputs, targets, quantiles=[0.5, 0.75, 0.9]):
    abs_errors = torch.abs(outputs - targets)
    metrics = {}
    for q in quantiles:
        tp = torch.quantile(abs_errors, q, dim=0).mean().item()
        metrics[f"TP{int(q*100)}"] = tp
    return metrics

train_dir = 'data/dataset/train'
val_dir = 'data/dataset/val'
test_dir = 'data/dataset/test'

# Paths to data directories and metadata files
train_images_dir = os.path.join(train_dir, 'images')
train_masks_dir = os.path.join(train_dir, 'masks')
train_metadata_file = os.path.join(train_dir, 'metadata.csv')

val_images_dir = os.path.join(val_dir, 'images')
val_masks_dir = os.path.join(val_dir, 'masks')
val_metadata_file = os.path.join(val_dir, 'metadata.csv')

test_images_dir = os.path.join(test_dir, 'images')
test_masks_dir = os.path.join(test_dir, 'masks')
test_metadata_file = os.path.join(test_dir, 'metadata.csv')

columns_list = [
    'Ankle Circumference (mm)',
    'Arm Length (Shoulder to Wrist) (mm)',
    'Arm Length (Shoulder to Elbow) (mm)', 
    'Armscye Circumference (Scye Circ Over Acromion) (mm)',
    'Chest Circumference (mm)',
    'Crotch Height (mm)',
    'Head Circumference (mm)',
    'Hip Circumference, Maximum (mm)',
    'Hip Circ Max Height (mm)',
    # 'Interscye Dst Stand (mm)',
    'Knee Height (mm)',
    'Neck Base Circumference (mm)',
    'Shoulder Breadth (mm)',
    'Thigh Circumference (mm)',
    'Waist Circumference, Pref (mm)',
    'Waist Height, Preferred (mm)'
]

# Define transformations
transform = T.Compose([
    T.Resize((640, 480)),
    T.ToTensor()
])

get_weight = False
get_gender = False

# Create datasets
train_dataset = BodyMeasurementDataset(train_dir, columns_list, transform, get_weight=get_weight, get_gender=get_gender)
val_dataset = BodyMeasurementDataset(val_dir, columns_list, transform, get_weight=get_weight, get_gender=get_gender)
test_dataset = BodyMeasurementDataset(test_dir, columns_list, transform, get_weight=get_weight, get_gender=get_gender)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)#, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)#, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)#, num_workers=4)

for inputs, targets in train_loader:
    print(f'Inputs of shape {inputs.shape}')
    break

# Instantiate the network and print its architecture
model = MeasureNet(num_outputs=len(columns_list), in_channels=inputs.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for param in model.parameters():
    param.requires_grad = False

for param in model.mlp.parameters():
    param.requires_grad = True

print_freq = 10

# Training loop
num_epochs = 10
metrics = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_rmse": [],
    "train_mae": [],
    "train_explained_variance": [],
    "train_r2": [],
    "val_rmse": [],
    "val_mae": [],
    "val_explained_variance": [],
    "val_r2": [],
    "train_TP50": [],
    "train_TP75": [],
    "train_TP90": [],
    "val_TP50": [],
    "val_TP75": [],
    "val_TP90": []
}

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_loss = 0.0
    train_rmse, train_mae, train_explained_variance, train_r2 = 0.0, 0.0, 0.0, 0.0
    train_tp_metrics = {"TP50": 0.0, "TP75": 0.0, "TP90": 0.0}
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if check_for_nans(i, inputs, "inputs") or check_for_nans(i, targets, "targets"):
            continue

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()
        batch_loss += loss.item()

        batch_rmse, batch_mae, batch_explained_variance, batch_r2 = calculate_metrics(outputs, targets)
        train_rmse += batch_rmse
        train_mae += batch_mae
        train_explained_variance += batch_explained_variance
        train_r2 += batch_r2

        batch_tp_metrics = quantile_metrics(outputs, targets)
        for key in train_tp_metrics:
            train_tp_metrics[key] += batch_tp_metrics[key]

        if (i+1)%print_freq==0:
            print(f'Batch {i+1}/{len(train_loader)}, Loss: {batch_loss/print_freq:.4f}')
            batch_loss = 0.0
    
    train_rmse /= len(train_loader)
    train_mae /= len(train_loader)
    train_explained_variance /= len(train_loader)
    train_r2 /= len(train_loader)
    for key in train_tp_metrics:
        train_tp_metrics[key] /= len(train_loader)
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_rmse, val_mae, val_explained_variance, val_r2 = 0.0, 0.0, 0.0, 0.0
    val_tp_metrics = {"TP50": 0.0, "TP75": 0.0, "TP90": 0.0}
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if check_for_nans(i, inputs, "val inputs") or check_for_nans(i, targets, "val targets"):
                continue
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            batch_rmse, batch_mae, batch_explained_variance, batch_r2 = calculate_metrics(outputs, targets)
            val_rmse += batch_rmse
            val_mae += batch_mae
            val_explained_variance += batch_explained_variance
            val_r2 += batch_r2

            batch_tp_metrics = quantile_metrics(outputs, targets)
            for key in val_tp_metrics:
                val_tp_metrics[key] += batch_tp_metrics[key]

            if (i+1)%print_freq==0:
                print(f'Batch {i+1}/{len(train_loader)}')
        
    val_rmse /= len(val_loader)
    val_mae /= len(val_loader)
    val_explained_variance /= len(val_loader)
    val_r2 /= len(val_loader)
    for key in val_tp_metrics:
        val_tp_metrics[key] /= len(val_loader)
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
    
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


    # Save metrics
    metrics["epoch"].append(epoch+1)
    metrics["train_loss"].append(epoch_loss / len(train_loader))
    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["train_rmse"].append(train_rmse)
    metrics["train_mae"].append(train_mae)
    metrics["train_explained_variance"].append(train_explained_variance)
    metrics["train_r2"].append(train_r2)
    metrics["val_rmse"].append(val_rmse)
    metrics["val_mae"].append(val_mae)
    metrics["val_explained_variance"].append(val_explained_variance)
    metrics["val_r2"].append(val_r2)
    metrics["train_TP50"].append(train_tp_metrics["TP50"])
    metrics["train_TP75"].append(train_tp_metrics["TP75"])
    metrics["train_TP90"].append(train_tp_metrics["TP90"])
    metrics["val_TP50"].append(val_tp_metrics["TP50"])
    metrics["val_TP75"].append(val_tp_metrics["TP75"])
    metrics["val_TP90"].append(val_tp_metrics["TP90"])

# Save metrics to CSV file
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('training_metrics.csv', index=False)

# Save the model
if best_model_state:
    torch.save(best_model_state, 'best_measure_net_model.pth')