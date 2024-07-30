import matplotlib.pyplot as plt
import numpy as np

def plt_images(image, mask, pred, threshold=0.5):
    pred = (pred > threshold).float()
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

    plt.savefig('preview.jpg', format='jpg', bbox_inches='tight')

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

def transfer_weights(detectron2_state_dict, pytorch_model):
    pytorch_state_dict = pytorch_model.state_dict()
    
    # Adjust detectron2_state_dict keys to match PyTorch model keys
    adjusted_dict = {}
    for key, value in detectron2_state_dict.items():
        new_key = key.replace('backbone.bottom_up', 'backbone')
        new_key = new_key.replace('stem.conv1.norm', 'stem.0.1')
        new_key = new_key.replace('stem.conv1', 'stem.0.0')
        if new_key in pytorch_state_dict:
            adjusted_dict[new_key] = value
    
    # Transfer weights
    for name, param in adjusted_dict.items():
        if name in pytorch_state_dict:
            if param.shape == pytorch_state_dict[name].shape:
                pytorch_state_dict[name].copy_(param)
            else:
                print(f"Shape mismatch for {name}: {param.shape} vs {pytorch_state_dict[name].shape}")
        else:
            print(f"Layer {name} not found in PyTorch model.")
    
    pytorch_model.load_state_dict(pytorch_state_dict, strict=False)
