import torch
import os
from utils.utils import get_class_by_path
from cut.data import create_dataset as cut_create_dataset

from utils.early_stopping import EarlyStopping

# LOAD MODULE
def load_module_class(module):
    module_path = f"modules.{module}"
    ModuleClass = get_class_by_path(module_path)

    return ModuleClass

# LOAD MODEL
def load_model_class(model):
    model_path = f"models.{model}"
    ModelClass = get_class_by_path(model_path)

    return ModelClass

# LOAD DATASET
def load_dataset_class(dataloader):
    dataset_path = f"dataloaders.{dataloader}"
    DataLoader = get_class_by_path(dataset_path)

    return DataLoader


def load_ct_labelmaps_training_data(hparams):
    CTDatasetLoader = load_dataset_class(hparams.dataloader_ct_labelmaps)
    dataloader = CTDatasetLoader(hparams)
    train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps  = dataloader.train_dataloader()
    val_loader_ct_labelmaps = dataloader.val_dataloader()
    return train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps, val_loader_ct_labelmaps


def load_real_us_training_data(opt_cut):
    real_us_dataset = cut_create_dataset(opt_cut)  # create a dataset given opt.dataset_mode and other options
    dataset_real_us = real_us_dataset.dataset   
    real_us_train_loader = torch.utils.data.DataLoader(dataset_real_us, batch_size=opt_cut.batch_size, shuffle=True, drop_last=True)

    return real_us_train_loader, dataset_real_us

def load_real_us_gt_test_data(hparams):
    RealUSGTDatasetClass = load_dataset_class(hparams.dataloader_real_us_test)
    real_us_gt_testdataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_test)
    real_us_gt_test_dataloader = torch.utils.data.DataLoader(real_us_gt_testdataset, shuffle=False)

    real_us_stopp_crit_dataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_stopp_crit)
    real_us_stopp_crit_dataloader = torch.utils.data.DataLoader(real_us_stopp_crit_dataset, shuffle=False)

    return real_us_gt_test_dataloader, real_us_stopp_crit_dataloader


def create_early_stopping(hparams, exp_name, mode):
    if not os.path.exists(hparams.output_path):
        os.mkdir(hparams.output_path)

    if mode == 'valid':
        ckpt_save_path_model1 = f'{hparams.output_path}/best_checkpoint_seg_renderer_valid_loss_{exp_name}'
        ckpt_save_path_model2 = f'{hparams.output_path}/best_checkpoint_CUT_val_loss_{exp_name}'

    early_stopping = EarlyStopping(patience=hparams.early_stopping_patience, 
                                    ckpt_save_path_model1=ckpt_save_path_model1, 
                                    ckpt_save_path_model2=ckpt_save_path_model2, 
                                    verbose=True)

    return early_stopping


def add_online_augmentations(hparams, module, input, label):

    if hparams.seg_net_input_augmentations_noise_blur:
        input = module.rendered_img_random_transf(input)

    if hparams.seg_net_input_augmentations_rand_crop:
        state = torch.get_rng_state()
        input = module.rendered_img_masks_random_transf(input)
        torch.set_rng_state(state)
        label = module.rendered_img_masks_random_transf(label)

    return input, label

