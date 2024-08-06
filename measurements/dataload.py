import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

class BodyMeasurementDataset(Dataset):
    def __init__(self, root_dir, measurement_columns, transform=None, m_inputs=True, get_weight=False, get_gender=False):
        self.images_dir = os.path.join(root_dir, 'images')
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        self.measurement_columns = measurement_columns
        self.transform = transform
        self.m_inputs = m_inputs
        self.get_weight = get_weight
        self.get_gender = get_gender

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        frontal_image_path = os.path.join(self.images_dir, row['frontal_image'])
        lateral_image_path = os.path.join(self.images_dir, row['lateral_image'])
        frontal_image = self.load_image(frontal_image_path)
        lateral_image = self.load_image(lateral_image_path)
        
        if self.transform:
            frontal_image = self.transform(frontal_image)
            lateral_image = self.transform(lateral_image)

        images = torch.cat((frontal_image, lateral_image), dim=2)

        targets = torch.tensor(row[self.measurement_columns].values.astype(float), dtype=torch.float32)

        if self.m_inputs:
            measurements = torch.zeros((1,images.shape[1],images.shape[2])) + torch.tensor(row['Stature (mm)'])
            if self.get_weight:
                weight = torch.zeros((1,images.shape[1],images.shape[2])) + torch.tensor(row['Weight (kg)'])
                measurements = torch.cat([measurements, weight], dim=0)
            
            if self.get_gender:
                gender = row['Gender']
                if gender == 'Male':
                    gender_t = torch.zeros((1,images.shape[1],images.shape[2]))
                else:
                    gender_t = torch.ones((1,images.shape[1],images.shape[2]))
                measurements = torch.cat([measurements, gender_t], dim=0)
            
            inputs = torch.cat([images, measurements])

            return inputs, targets
        
        else:
            measurements = torch.tensor(row['Stature (mm)']).unsqueeze(-1)
            if self.get_weight:
                weight = torch.tensor(row['Weight (kg)']).unsqueeze(-1)
                measurements = torch.cat([measurements, weight], dim=0)
            
            if self.get_gender:
                gender = row['Gender']
                if gender == 'Male':
                    gender_t = torch.tensor(0).unsqueeze(-1)
                else:
                    gender_t = torch.tensor(1).unsqueeze(-1)
                measurements = torch.cat([measurements, gender_t], dim=0)
            
            return (images, measurements), targets

    def load_image(self, path):
        image = Image.open(path)
        new_image = Image.new('RGB', image.size, (255, 255, 255))
        new_image.paste(image, mask=image.split()[3]) 
        return new_image