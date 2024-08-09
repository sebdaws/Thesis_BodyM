import torch
import torch.nn as nn
from sklearn.metrics import explained_variance_score

def check_for_nans(i, tensor, name):
    if torch.isnan(tensor).any():
        # print(f"NaNs found in {name} {i+1}")
        return True
    return False

# Function to calculate additional metrics
def r2_score(outputs, targets):
    ss_res = torch.sum((targets - outputs) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def calculate_metrics(outputs, targets, criterion):
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