import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class PolygonDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None, is_train=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.input_dir = os.path.join(data_dir, 'inputs')
        self.output_dir = os.path.join(data_dir, 'outputs')
        
        # Color mapping
        self.color_map = {
            'cyan': 0, 'purple': 1, 'magenta': 2, 'green': 3, 'red': 4,
            'blue': 5, 'yellow': 6, 'orange': 7
        }
    
    def __len__(self):
        return len(self.data)
    
    def transform_pair(self, input_img, output_img):
        """Apply consistent transforms to both input and output images"""
        # Resize both images to target size (128x128 works better with UNet)
        input_img = TF.resize(input_img, (128, 128), interpolation=TF.InterpolationMode.BILINEAR)
        output_img = TF.resize(output_img, (128, 128), interpolation=TF.InterpolationMode.BILINEAR)
        
        if self.is_train:
            # Random rotation (same angle for both images)
            if random.random() > 0.5:
                angle = random.uniform(-90, 90)
                input_img = TF.rotate(input_img, angle)
                output_img = TF.rotate(output_img, angle)
            
            # Random horizontal flip (same for both)
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                output_img = TF.hflip(output_img)
            
            # Random vertical flip (same for both)  
            if random.random() > 0.5:
                input_img = TF.vflip(input_img)
                output_img = TF.vflip(output_img)
        
        # Convert to tensors
        input_tensor = TF.to_tensor(input_img)
        output_tensor = TF.to_tensor(output_img)
        
        return input_tensor, output_tensor
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        input_path = os.path.join(self.input_dir, entry['input_polygon'])
        output_path = os.path.join(self.output_dir, entry['output_image'])
        
        try:
            input_img = Image.open(input_path).convert('L')
            output_img = Image.open(output_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {input_path} or {output_path}: {e}")
            raise
        
        color = self.color_map[entry['colour']]
        
        # Apply transforms
        input_tensor, output_tensor = self.transform_pair(input_img, output_img)
        
        # Verify sizes
        assert input_tensor.shape[1:] == (128, 128), f"Input tensor has wrong size: {input_tensor.shape}"
        assert output_tensor.shape[1:] == (128, 128), f"Output tensor has wrong size: {output_tensor.shape}"
        
        return input_tensor, torch.tensor(color, dtype=torch.long), output_tensor

# Simple function to create datasets
def create_datasets(train_data_dir, train_json_file, val_data_dir, val_json_file):
    """Create training and validation datasets"""
    train_dataset = PolygonDataset(train_data_dir, train_json_file, is_train=True)
    val_dataset = PolygonDataset(val_data_dir, val_json_file, is_train=False)
    return train_dataset, val_dataset