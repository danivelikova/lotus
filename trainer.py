import time
import torch, torchvision
import wandb
import numpy as np
from tqdm.auto import tqdm
from cut.cut_trainer import CUTTrainer
import helpers
from models.us_rendering_model import UltrasoundRendering


class Trainer:
    def __init__(self, hparams, opt_cut, plotter):
        self.hparams = hparams
        self.opt_cut = opt_cut
        self.plotter = plotter

        ModuleClass = helpers.load_module_class(hparams.module)
        InnerModelClass = helpers.load_model_class(hparams.inner_model)
        self.inner_model = InnerModelClass(params=hparams)
        
        if not hparams.outer_model_monai: 
            OuterModelClass = helpers.load_model_class(hparams.outer_model)
            outer_model = OuterModelClass(hparams=hparams)
            self.module = ModuleClass(params=hparams, inner_model=self.inner_model, outer_model=outer_model)
        else:
            self.module = ModuleClass(params=hparams, inner_model=self.inner_model)

        self.real_us_train_loader, dataset_real_us = helpers.load_real_us_training_data(opt_cut)
        self.cut_trainer = CUTTrainer(opt_cut, dataset_real_us, self.real_us_train_loader)
        self.USRendereDefParams = UltrasoundRendering(params=hparams, default_param=True).to(hparams.device)

# ------------------------------------------------------------------------------------------------------------------------------
#                                         Train US Renderer + SEG NET + CUT
# ------------------------------------------------------------------------------------------------------------------------------
    def train_step(self, train_loader_ct_labelmaps, epoch):
        self.module.train()
        self.module.outer_model.train()
        self.inner_model.train()

        iter_data_time = time.time()    # timer for data loading per iteration
        dataloader_real_us_iterator = iter(self.real_us_train_loader)
        train_losses = []
        for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps), ncols= 100):
            self.module.optimizer.zero_grad()
            input, label, _ = self.module.get_data(batch_data_ct)

            # if 1 in label:  
            if (label != 0).any():  # if not empty label
                us_rendr = self.module.rendering_forward(input)

                us_rendr_cut = us_rendr.clone().detach()
                self.cut_trainer.train_cut(self.module, us_rendr_cut, epoch, dataloader_real_us_iterator, iter_data_time)
                
                #if using the identity image from CUT instead of us_rendr directly
                if self.hparams.use_idtB:  
                    idt_B = self.cut_trainer.get_idtB(us_rendr)
                    us_rendr = idt_B

                #add augmentations during training
                if self.hparams.seg_net_input_augmentations_noise_blur or self.hparams.seg_net_input_augmentations_rand_crop:
                    us_rendr, label = helpers.add_online_augmentations(self.hparams, self.module, us_rendr, label)

                loss, _ = self.module.seg_forward(us_rendr, label)

                if self.hparams.inner_model_learning_rate == 0: 
                    self.cut_trainer.cut_model.set_requires_grad(self.module.USRenderingModel, False)
                
                loss.backward()

                # Perform gradient clipping
                if self.hparams.grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.module.USRenderingModel.parameters(), max_norm=1)
                    torch.nn.utils.clip_grad_norm_(self.module.outer_model.parameters(), max_norm=1)

                self.module.optimizer.step()

                if self.hparams.logging: wandb.log({"train_loss_step": loss.item()})
                train_losses.append(loss.item())
                if self.hparams.logging: self.plotter.log_us_rendering_values(self.module.USRenderingModel, i)
                    

            if self.hparams.scheduler: 
                self.module.scheduler.step()
                wandb.log({"lr_": self.module.optimizer.param_groups[0]["lr"]})
                wandb.log({"lr_2": self.module.optimizer.param_groups[1]["lr"]})

            # Plot CUT results during training
            if len(self.cut_trainer.cut_plot_figs) > 0: 
                self.plotter.log_image(torchvision.utils.make_grid(self.cut_trainer.cut_plot_figs), "real_A|fake_B|real_B|idt_B")
                self.cut_trainer.cut_plot_figs = []
        
        train_loss = np.average(train_losses)
        return train_loss



    def evaluate(self, val_loader_ct_labelmaps, epoch):

        self.module.eval()
        self.module.outer_model.eval()
        self.inner_model.eval()
        self.cut_trainer.cut_model.eval()
        self.USRendereDefParams.eval()
        def_renderer_plot_figs = []
        print(f"--------------- VALIDATION SEG NET ------------")
        valid_losses = []
        with torch.no_grad():
            for nr, val_batch_data_ct in tqdm(enumerate(val_loader_ct_labelmaps), total=len(val_loader_ct_labelmaps), ncols= 100):
                
                val_input, val_label, filename = self.module.get_data(val_batch_data_ct)  

                if (val_label != 0).any():
                    val_input_copy =  val_input.clone().detach()    
                    us_rendr_val = self.module.rendering_forward(val_input)   

                    if self.hparams.use_idtB:
                        idt_B_val = self.cut_trainer.get_idtB(us_rendr_val)
                        us_rendr_val = idt_B_val

                    val_loss_step, rendered_seg_pred = self.module.seg_forward(us_rendr_val, val_label)

                    valid_losses.append(val_loss_step.item())

                    if self.hparams.log_default_renderer and nr < self.hparams.nr_imgs_to_plot:
                        us_rendr_def = self.USRendereDefParams(val_input_copy.squeeze()) 
                        if not self.hparams.use_idtB: idt_B_val = us_rendr_val

                        plot_fig = self.plotter.plot_stopp_crit(caption="labelmap|default_renderer|learnedUS|seg_input|seg_pred|gt",
                                                    imgs=[val_input, us_rendr_def, us_rendr_val, rendered_seg_pred, val_label], 
                                                    img_text='', epoch=epoch, plot_single=False)
                        def_renderer_plot_figs.append(plot_fig)

                    elif not self.hparams.log_default_renderer:
                        dict = self.module.plot_val_results(val_input, val_loss_step, filename, val_label, rendered_seg_pred, us_rendr_val, epoch)
                        self.plotter.validation_batch_end(dict)

            if len(def_renderer_plot_figs) > 0: 
                if self.hparams.logging:
                    self.plotter.log_image(torchvision.utils.make_grid(def_renderer_plot_figs), "default_renderer|labelmap|defaultUS|learnedUS|seg_pred|gt")

        avg_valid_loss = np.average(valid_losses)
        print(f"--------------- END SEG NETWORK VALIDATION ------------")
        return avg_valid_loss

    # ------------------------------------------------------------------------------------------------------------------------------
    #    (GT STOPPING CRITERION) INFER REAL_US_STOPP_CRIT_TEST_SET_IMGS THROUGH THE CUT+SEG NET for stopping the end2end training
    # -------------------------------------------------------------------------------------------------------------------------------
    def stopp_crit_check(self, real_us_stopp_crit_dataloader, epoch, early_stopping):
        stoppcrit_losses = []
        self.cut_trainer.cut_model.eval()
        avg_stopp_crit_loss = 1
        if self.hparams.stopping_crit_gt_imgs and epoch > self.hparams.epochs_check_stopp_crit:
            print(f"--------------- STOPPING CRITERIA US IMGS CHECK ------------")
            stopp_crit_imgs_plot_figs = []
            with torch.no_grad():
                for nr, batch_data_real_us_stopp_crit in tqdm(enumerate(real_us_stopp_crit_dataloader), total=len(real_us_stopp_crit_dataloader), ncols= 100, position=0, leave=True):
                    
                    real_us_stopp_crit_img, real_us_stopp_crit_label = batch_data_real_us_stopp_crit[0].to(self.hparams.device), batch_data_real_us_stopp_crit[1].to(self.hparams.device).float()
                    
                    reconstructed_us_stopp_crit = self.cut_trainer.cut_model.netG(real_us_stopp_crit_img)
                    reconstructed_us_stopp_crit = (reconstructed_us_stopp_crit / 2 ) + 0.5 # from [-1,1] to [0,1]

                    stoppcrit_loss, seg_pred_stopp_crit  = self.module.seg_forward(reconstructed_us_stopp_crit, real_us_stopp_crit_label)
                    stoppcrit_losses.append(stoppcrit_loss)

                    if self.hparams.logging:
                        real_us_stopp_crit_img = (real_us_stopp_crit_img / 2 ) + 0.5 # from [-1,1] to [0,1]
                        wandb.log({"stoppcrit_loss": stoppcrit_loss.item()})
                        plot_fig_gt = self.plotter.plot_stopp_crit(caption="stopp_crit|real_us|reconstructed_us|seg_pred|gt_label",
                                                imgs=[real_us_stopp_crit_img, reconstructed_us_stopp_crit, seg_pred_stopp_crit, real_us_stopp_crit_label], 
                                                img_text='loss=' + "{:.4f}".format(stoppcrit_loss.item()), epoch=epoch, plot_single=False)
                        stopp_crit_imgs_plot_figs.append(plot_fig_gt)
            
            if len(stopp_crit_imgs_plot_figs) > 0: 
                self.plotter.log_image(torchvision.utils.make_grid(stopp_crit_imgs_plot_figs), "stopp_crit|real_us|reconstructed_us|seg_pred|gt_label")
                avg_stopp_crit_loss = torch.mean(torch.stack(stoppcrit_losses))
                wandb.log({"stoppcrit_loss_epoch": avg_stopp_crit_loss, "epoch": epoch})
                stoppcrit_losses = []
                stopp_crit_imgs_plot_figs = [] 
            
            if epoch > self.hparams.min_epochs:
                early_stopping(avg_stopp_crit_loss, epoch, self.module, self.cut_trainer.cut_model.netG)
            

                    