from PIL import Image
import os
from torch.utils.data import Dataset
#test
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, im_transform=None, mask_transform=None):
        """
        Args:
            root_dir: Directory with multiple subdirectories, each containing 'images' and 'masks' subdirectories.
            im_transform: Transform to be applied on images.
            mask_transform: Transform to be applied on masks.
        """
        self.im_transform = im_transform
        self.mask_transform = mask_transform
        self.images = []  # List to store image paths
        self.masks = []   # List to store corresponding mask paths

        # Iterate over each subfolder in the root directory
        for subdir in os.listdir(root_dir):
            image_dir = os.path.join(root_dir, subdir, 'images')
            mask_dir = os.path.join(root_dir, subdir, 'masks')

            # Check if 'images' and 'masks' subdirectories exist
            if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
                # List each image and corresponding mask path
                for img_file in os.listdir(image_dir):
                    img_path = os.path.join(image_dir, img_file)
                    mask_path = os.path.join(mask_dir, img_file)
                    if os.path.isfile(img_path) and os.path.isfile(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Paths for the image and mask
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        # Load the image and mask files
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        # Apply transform to both image and mask if a transform is specified
        if self.im_transform:
            image = self.im_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
