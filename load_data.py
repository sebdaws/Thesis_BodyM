from PIL import Image
import os
from torch.utils.data import Dataset

class SegmentationDataset_old(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with images and mask subdirectories.
            transform: Optional transform to be applied on a sample.
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # convert mask to grayscale

        if self.transform:
            image, mask = self.transform(image), self.transform(mask)

        return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with multiple subdirectories, each containing 'images' and 'masks' subdirectories.
            transform: Optional transform to be applied on a sample.
        """
        self.transform = transform
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
        mask = Image.open(mask_name).convert("L")  # convert mask to grayscale

        # Apply transform to both image and mask if a transform is specified
        if self.transform:
            image, mask = self.transform(image), self.transform(mask)

        return image, mask
