import random
import wandb
import torch
from tqdm.auto import trange
import configargparse
from utils.plotter import Plotter
from utils.configargparse_arguments import build_configargparser
from utils.utils import argparse_summary
from cut.lotus_options import LOTUSOptions
import helpers
import trainer

MANUAL_SEED = False


if __name__ == "__main__":

    if MANUAL_SEED: torch.manual_seed(2023)
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_cut = LOTUSOptions().parse()   # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

    # no horizontal flip for aorta_only segmentation task
    if hparams.aorta_only: opt_cut.no_flip = True

    if hparams.debug:
        hparams.exp_name = 'DEBUG'
    else:
        hparams.exp_name += str(hparams.pred_label) + '_' + str(opt_cut.lr) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(hparams.outer_model_learning_rate)
    
    hparams.exp_name = str(random.randint(0, 1000)) + "_" + hparams.exp_name

    if hparams.logging: wandb.init(name=hparams.exp_name, project=hparams.project_name) #, silent=True)
    plotter = Plotter()

    argparse_summary(hparams, parser)
    # ---------------------
    # LOAD DATA 
    # ---------------------
    train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps, val_loader_ct_labelmaps = helpers.load_ct_labelmaps_training_data(hparams)
    _, real_us_stopp_crit_dataloader = helpers.load_real_us_gt_test_data(hparams)

    early_stopping = helpers.create_early_stopping(hparams, hparams.exp_name, 'test')
    early_stopping_best_val = helpers.create_early_stopping(hparams, hparams.exp_name, 'valid')

    avg_train_losses, avg_valid_losses = ([] for i in range(2))

    # --------------------
    # RUN TRAINING
    # ---------------------

    trainer = trainer.Trainer(hparams, opt_cut, plotter)  # Initialize the Trainer
    trainer.cut_trainer.cut_model.set_requires_grad(trainer.USRendereDefParams, False)

    for epoch in trange(1, hparams.max_epochs + 1):

        train_loss = trainer.train_step(train_loader_ct_labelmaps, epoch)
        # calculate average loss over an epoch
        avg_train_losses.append(train_loss)

        # SEG NET VALIDATION on every hparams.validate_every_n_steps
        if epoch % hparams.validate_every_n_steps == 0:
            valid_loss = trainer.evaluate(val_loader_ct_labelmaps, epoch)
            avg_valid_losses.append(valid_loss)
            
        epoch_len = len(str(hparams.max_epochs))
        print(f'[{epoch:>{epoch_len}}/{hparams.max_epochs:>{epoch_len}}] ' +
                    f'seg_train_loss_epoch: {train_loss:.5f} ' +
                    f'seg_valid_loss_epoch: {valid_loss:.5f}')

        if hparams.logging: 
            wandb.log({"seg_train_loss_epoch": train_loss, "epoch": epoch})
            wandb.log({"seg_val_loss_epoch": valid_loss, "epoch": epoch})
            if not hparams.log_default_renderer: 
                plotter.validation_epoch_end()

        trainer.stopp_crit_check(real_us_stopp_crit_dataloader, epoch, early_stopping)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print(f'train completed, avg_train_losses: {avg_train_losses} avg_valid_losses: {avg_valid_losses}')
    print(f'best val_loss: {early_stopping.val_loss_min} at best_epoch: {early_stopping.best_epoch}')