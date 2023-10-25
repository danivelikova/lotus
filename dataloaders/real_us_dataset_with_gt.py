import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torch.utils.data import random_split
import torchvision.transforms as transforms

SIZE_W = 256
SIZE_H = 256

class RealUSGTDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir_imgs = root_dir + 'imgs/'
        self.root_dir_masks = root_dir + 'masks/'
        self.transform_img = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.NEAREST)
        ])

        self.image_files = [f for f in sorted(os.listdir(os.getcwd() + '/' + self.root_dir_imgs)) if f.endswith('.jpg') or f.endswith('.png')]
        self.masks_files = [f for f in sorted(os.listdir(self.root_dir_masks)) if f.endswith('.jpg') or f.endswith('.png')]
        print("len(self.image_files): ", len(self.image_files))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir_imgs, self.image_files[idx])
        image = Image.open(image_path)#.convert('L')

        mask_path = os.path.join(self.root_dir_masks, self.masks_files[idx])
        mask = np.asarray(Image.open(mask_path))

        image = self.transform_img(image)
        
        mask = self.transform_mask(mask)
        mask = torch.where(mask > 0 , 1, 0)

        return image, mask


class RealUSGTDataLoader():
    def __init__(self, param, root_dir, batch_size=1, transform=None, num_workers=1):
        super().__init__()
        self.params = param
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def get_dataloaders(self):
        dataset = RealUSGTDataset(root_dir=self.root_dir, transform=self.transform)
       
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


        return train_loader, val_loader