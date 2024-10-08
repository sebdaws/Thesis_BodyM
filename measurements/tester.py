import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import argparse
import os
import pandas as pd 

from dataload import BodyMeasurementDataset
from build_model import MeasureNet, MeasureViT
from utils import check_for_nans, calculate_metrics, quantile_metrics

def measurement_inputs(args, all_measurements):
    measurements = all_measurements[:, 0].unsqueeze(1)
    if args.weight:
        measurements = torch.cat([measurements, all_measurements[:, 1].unsqueeze(1)], dim=1)
    if args.gender:
        measurements = torch.cat([measurements, all_measurements[:, 2].unsqueeze(1)], dim=1)
    return measurements

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='Path of the model to test.')
parser.add_argument('--batch_size', required=False, type=int, default=8, help='Size of batch for training. (default is 8)')
parser.add_argument('--weight', required=False, action='store_true', help='Specify wether to use the weight as input.')
parser.add_argument('--gender', required=False, action='store_true', help='Specify wether to use the gender as input.')
parser.add_argument('--m_inputs', required=False, action='store_false', help='Specify wether additional measurements are added at backbone input or mlp input (i.e. after feature extraction).')
parser.add_argument('--vit', required=False, action='store_true', help='Specify whether to use ViTPose.')
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

if  args.vit:
    transform = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor()
    ])
else:
    transform = T.Compose([
        T.Resize((640, 480)),
        T.ToTensor()
    ])
print('Loading data...')
# Create datasets
if args.vit:
    concat = 'channel'
else:
    concat = 'width'
test_dataset = BodyMeasurementDataset(test_dir, columns_list, transform, m_inputs=args.m_inputs, concat=concat, get_weight=args.weight, get_gender=args.gender)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)#, num_workers=4)

print(f'Loaded Data, {len(test_dataset)} samples')

for inputs, targets in test_loader:
    if args.m_inputs:
        images = inputs
        num_m = images.shape[1]-3
    else:
        images, all_measurements = inputs
        print(all_measurements.shape)
        measurements = measurement_inputs(args, all_measurements)
        print(measurements.shape)
        num_m = measurements.shape[1]
    print(f'Inputs of shape {images.shape}')
    break

if args.vit:
    print('Loading model: Vision Transformer')
    vitpose_name = 'ViTPose_base_coco_256x192'
    model = MeasureViT(num_outputs=len(columns_list), vitpose_name=vitpose_name, vitpose_path=None, num_m=num_m).to(device)
else:
    print('Loading model: MNasNet')
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

model_id = os.path.basename(args.model_path).replace('.pt', '')

gender_metrics = {
    'male': {'model_id': model_id, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'exp_var': 0.0, 'r2': 0.0, 'TP50': 0.0, 'TP75': 0.0, 'TP90': 0.0, 'count': 0},
    'female': {'model_id': model_id, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'exp_var': 0.0, 'r2': 0.0, 'TP50': 0.0, 'TP75': 0.0, 'TP90': 0.0, 'count': 0}
}

class_metrics = {col: {'model_id': model_id, 'mae': 0.0, 'TP50': 0.0, 'TP75': 0.0, 'TP90': 0.0} for col in columns_list}

print(f'Testing {args.model_path}')
with torch.no_grad():
    print('Performing Testing...')
    for i, (inputs, targets) in enumerate(test_loader):
        if args.m_inputs:
            images = inputs
            measurements = None
        else:
            images, all_measurements = inputs
            measurements = measurement_inputs(args, all_measurements)
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
        
        # Calculate class-wise metrics
        for j, col in enumerate(columns_list):
            class_mae = nn.L1Loss()(outputs[:, j], targets[:, j]).item()

            class_metrics[col]['mae'] += class_mae

            quantiles = quantile_metrics(outputs[:, j].unsqueeze(1), targets[:, j].unsqueeze(1))
            for key in ['TP50', 'TP75', 'TP90']:
                class_metrics[col][key] += quantiles[key]

        if args.m_inputs==False:
            genders = all_measurements[:, -1]  # Assuming the last measurement column is the gender
            for gender in [0, 1]:  # 0 for male, 1 for female
                gender_mask = (genders == gender)
                gender_str = 'male' if gender == 0 else 'female'
                if gender_mask.sum() > 0:
                    gender_outputs = outputs[gender_mask]
                    gender_targets = targets[gender_mask]

                    gender_loss = criterion(gender_outputs, gender_targets)
                    gender_rmse, gender_mae, gender_exp_var, gender_r2 = calculate_metrics(gender_outputs, gender_targets, criterion)
                    gender_tp_metrics = quantile_metrics(gender_outputs, gender_targets)

                    gender_metrics[gender_str]['mse'] += gender_loss.item()
                    gender_metrics[gender_str]['rmse'] += gender_rmse
                    gender_metrics[gender_str]['mae'] += gender_mae
                    gender_metrics[gender_str]['exp_var'] += gender_exp_var
                    gender_metrics[gender_str]['r2'] += gender_r2
                    for key in ['TP50', 'TP75', 'TP90']:
                        gender_metrics[gender_str][key] += gender_tp_metrics[key]
                    gender_metrics[gender_str]['count'] += 1

mse /= len(test_loader)
rmse /= len(test_loader)
mae /= len(test_loader)
exp_var /= len(test_loader)
r2 /= len(test_loader)
for key in tp_metrics:
    tp_metrics[key] /= len(test_loader)

for col in columns_list:
    for metric in ['mae', 'TP50', 'TP75', 'TP90']:
        class_metrics[col][metric] /= len(test_loader)

for gender in ['male', 'female']:
    count = gender_metrics[gender]['count']
    if count > 0:
        gender_metrics[gender]['mse'] /= count
        gender_metrics[gender]['rmse'] /= count
        gender_metrics[gender]['mae'] /= count
        gender_metrics[gender]['exp_var'] /= count
        gender_metrics[gender]['r2'] /= count
        for key in ['TP50', 'TP75', 'TP90']:
            gender_metrics[gender][key] /= count

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

# Save class-wise metrics to separate CSV
class_metrics_df = pd.DataFrame(class_metrics).T.reset_index()
class_metrics_df.rename(columns={'index': 'Measurement'}, inplace=True)
class_metrics_df.set_index(['model_id', 'Measurement'], inplace=True)
class_metrics_path = os.path.join('test_metrics', 'class_metrics.csv')

if not os.path.isfile(class_metrics_path):
    class_metrics_df.to_csv(class_metrics_path, index=True)
else:
    class_metrics_df.to_csv(class_metrics_path, mode='a', header=False, index=True)

print(f'Class-wise metrics saved to {class_metrics_path}')

gender_metrics_df = pd.DataFrame(gender_metrics).T.reset_index()
gender_metrics_df.rename(columns={'index': 'Gender'}, inplace=True)
gender_metrics_df.set_index(['model_id', 'Gender'], inplace=True)
gender_metrics_path = os.path.join('test_metrics', 'gender_metrics.csv')

if not os.path.isfile(gender_metrics_path):
    gender_metrics_df.to_csv(gender_metrics_path, index=True)
else:
    gender_metrics_df.to_csv(gender_metrics_path, mode='a', header=False, index=True)

print(f'Gender-specific metrics saved to {gender_metrics_path}')