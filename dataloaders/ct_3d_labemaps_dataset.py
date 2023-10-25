import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import nibabel as nib
from torch.utils.data import random_split
import torchvision.transforms as transforms


SIZE_W = 256
SIZE_H = 256
class CT3DLabelmapDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.n_classes = params.n_classes

        self.base_folder_data_imgs = params.base_folder_data_path
        self.base_folder_data_masks = params.base_folder_mask_path
        self.labelmap_path = params.labelmap_path

        self.sub_folder_CT = [sub_f for sub_f in sorted(os.listdir(os.getcwd() + '/' + self.base_folder_data_imgs))]
        self.full_labelmap_path_imgs = [self.base_folder_data_imgs + s + self.labelmap_path for s in self.sub_folder_CT]
        self.full_labelmap_path_masks = [self.base_folder_data_masks + s + self.labelmap_path for s in self.sub_folder_CT]

        self.slice_indices, self.volume_indices, self.total_slices, self.volumes = self.read_volumes(self.full_labelmap_path_imgs)
        self.mask_slice_indices, self.mask_volume_indices, self.mask_total_slices, self.mask_volumes = self.read_volumes(self.full_labelmap_path_masks)


        if self.params.aorta_only:
            # self.transform_img = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.RandomAffine(degrees=(0, 30), translate=(0.2, 0.2), scale=(1.0, 2.0), fill=9),
            #     transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.NEAREST),
            # ])
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([380, 380], transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop((SIZE_W)),
            ])
        else:
            # self.transform_img = transforms.Compose([
            #     transforms.ToTensor(),
            #     # transforms.RandomAffine(degrees=(0, 30), translate=(0.2, 0.2), scale=(1.0, 2.0), fill=9),
            #     # transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.NEAREST),
            #     # transforms.RandomVerticalFlip()
            # ])
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([380, 380], transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop((SIZE_W)),
            ])

        if self.params.pred_label == 13:    # spine
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=(0, 30), translate=(0.2, 0.2), scale=(0.9, 1.0), fill=9),
                transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.NEAREST),
            ])


    def __len__(self):
        if self.params.debug:
            return self.total_slices  // 20     #reduce dataset size for debugging
        else:
            return self.total_slices 

    def read_volumes(self, full_labelmap_path):
        slice_indices = []
        volume_indices = []
        total_slices = 0
        volumes = []

        for idx, folder in enumerate(full_labelmap_path):
            labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]
            vol_nib = nib.load(folder + labelmap)
            vol = vol_nib.get_fdata()

            slice_indices.extend(np.arange(vol.shape[2]))  #append the vol indexes
            volume_indices.extend(np.full(shape=vol.shape[2], fill_value=idx, dtype=np.int32))  #append the vol indexes
            total_slices += vol.shape[2]
            volumes.append(vol)
        
        return slice_indices, volume_indices, total_slices, volumes


    def preprocess(self, img, mask):
        if mask:
            img = np.where(img != self.params.pred_label, 0, 1)
            
        return img 

    
    def __getitem__(self, idx):

        vol_nr = self.volume_indices[idx]
        labelmap_slice = self.volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')        #labelmap input to the US renderer
        if self.full_labelmap_path_imgs != self.base_folder_data_masks:
            mask_slice = self.mask_volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')
        else:
            mask_slice = labelmap_slice

        state = torch.get_rng_state()
        labelmap_slice = self.transform_img(labelmap_slice)
        torch.set_rng_state(state)
        mask_slice = self.transform_img(mask_slice)

        mask_slice = torch.where(mask_slice != self.params.pred_label, 0, 1)
        
        # for spine we flip the labelmap horizontally
        if self.params.pred_label == 13:
            labelmap_slice = transforms.functional.hflip(labelmap_slice)
            mask_slice = transforms.functional.hflip(mask_slice)

        # for aorta_only
        if self.params.aorta_only:
            labelmap_slice = transforms.functional.vflip(labelmap_slice)
            mask_slice = transforms.functional.vflip(mask_slice)

        return labelmap_slice, mask_slice, str(vol_nr) + '_' + str(self.slice_indices[idx])


class CT3DLabelmapDataLoader():
    def __init__(self, params):
        super().__init__()
        self.params = params

    def train_dataloader(self):
        full_dataset = CT3DLabelmapDataset(self.params) 
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        
        return train_loader, self.train_dataset, self.val_dataset 

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers)
