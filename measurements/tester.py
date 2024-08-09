import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import argparse
import os
import pandas as pd 

from dataload import BodyMeasurementDataset
from build_model import MeasureNet
from utils import check_for_nans, calculate_metrics, quantile_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='Path of the model to test.')
parser.add_argument('--batch_size', required=False, type=int, default=8, help='Size of batch for training. (default is 8)')
parser.add_argument('--weight', required=False, action='store_true', help='Specify wether to use the weight as input.')
parser.add_argument('--gender', required=False, action='store_true', help='Specify wether to use the gender as input.')
parser.add_argument('--m_inputs', required=False, action='store_false', help='Specify wether additional measurements are added at backbone input or mlp input (i.e. after feature extraction).')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_dir = 'data/dataset/test'

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

test_dataset = BodyMeasurementDataset(test_dir, columns_list, transform, m_inputs=args.m_inputs, get_weight=args.weight, get_gender=args.gender)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)#, num_workers=4)

print(f'Loaded Data, {len(test_dataset)} samples')

for inputs, targets in test_loader:
    if args.m_inputs:
        images = inputs
        num_m = images.shape[1]-3
    else:
        images, measurements = inputs
        num_m = measurements.shape[1]
    print(f'Inputs of shape {images.shape}')
    break

model = MeasureNet(num_outputs=len(columns_list), num_m=num_m, m_inputs=args.m_inputs).to(device)

try:
    model.load_state_dict(torch.load(args.model_path))
except:
    raise ValueError('Make sure the arguments match the model structure.')

print('Loaded Model with weights')

criterion = nn.MSELoss()

model.eval()
mse = 0.0
rmse, mae, exp_var, r2 = 0.0, 0.0, 0.0, 0.0
tp_metrics = {"TP50": 0.0, "TP75": 0.0, "TP90": 0.0}
print(f'Testing {args.model_path}')
with torch.no_grad():
    print('Performing Testing...')
    for i, (inputs, targets) in enumerate(test_loader):
        if args.m_inputs:
            images = inputs
            measurements = None
        else:
            images, measurements = inputs
            measurements = measurements.to(device)
            if check_for_nans(i, measurements, 'val measurements'):
                continue
        images = images.to(device)
        targets = targets.to(device)

        if check_for_nans(i, images, "val inputs") or check_for_nans(i, targets, "val targets"):
            continue
        outputs = model(images, measurements)
        loss = criterion(outputs, targets)
        mse += loss.item()

        batch_rmse, batch_mae, batch_explained_variance, batch_r2 = calculate_metrics(outputs, targets, criterion)
        rmse += batch_rmse
        mae += batch_mae
        exp_var += batch_explained_variance
        r2 += batch_r2

        batch_tp_metrics = quantile_metrics(outputs, targets)
        for key in tp_metrics:
            tp_metrics[key] += batch_tp_metrics[key]

mse /= len(test_loader)
rmse /= len(test_loader)
mae /= len(test_loader)
exp_var /= len(test_loader)
r2 /= len(test_loader)
for key in tp_metrics:
    tp_metrics[key] /= len(test_loader)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'Exp Var: {exp_var:.4f}')
print(f'R2: {r2:.4f}')
for key in tp_metrics:
    print(f'{key}: {tp_metrics[key]:.4f}')

# Save metrics to CSV
metrics = {
    'model': args.model_path,
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'Exp_Var': exp_var,
    'R2': r2,
    'TP50': tp_metrics['TP50'],
    'TP75': tp_metrics['TP75'],
    'TP90': tp_metrics['TP90']
}

# Convert to DataFrame
metrics_df = pd.DataFrame([metrics])

# Create directory if it doesn't exist
os.makedirs('test_metrics', exist_ok=True)

# Save to CSV
csv_path = os.path.join('test_metrics', 'metrics.csv')
if not os.path.isfile(csv_path):
    metrics_df.to_csv(csv_path, index=False)
else:
    metrics_df.to_csv(csv_path, mode='a', header=False, index=False)

print(f'Metrics saved to {csv_path}')