import torch
import time
import wandb
from cut.data.base_dataset import get_transform as cut_get_transform
from cut.models import create_model as cut_create_model
from utils.plotter import Plotter

class CUTTrainer():
    def __init__(self, opt_cut, dataset_real_us, real_us_train_loader):
        super(CUTTrainer, self).__init__()
        self.opt_cut = opt_cut
        self.t_data = 0 
        self.total_iter = 0 
        self.optimize_time = 0.1
        self.dataset_real_us = dataset_real_us
        self.random_state = None
        self.real_us_train_loader = real_us_train_loader
        self.init_cut = True
        self.cut_plot_figs = []
        self.data_cut_real_us = []
        self.plotter = Plotter()
        self.cut_model = cut_create_model(opt_cut)   

    def cut_optimize_parameters(self, module, dice_loss):
        # update Discriminator
        self.cut_model.set_requires_grad(self.cut_model.netD, True)
        self.cut_model.optimizer_D.zero_grad()
        self.cut_model.loss_D = self.cut_model.compute_D_loss()
        self.cut_model.loss_D.backward()
        self.cut_model.optimizer_D.step()

        # update Generator
        self.cut_model.set_requires_grad(self.cut_model.netD, False)
        self.cut_model.optimizer_G.zero_grad()
        if self.cut_model.opt.netF == 'mlp_sample':
            self.cut_model.optimizer_F.zero_grad()
        self.cut_model.loss_G = self.cut_model.compute_G_loss()
        
        self.cut_model.loss_G.backward()

    def cut_optimizer_step(self):
        self.cut_model.optimizer_G.step()
        if self.cut_model.opt.netF == 'mlp_sample':
            self.cut_model.optimizer_F.step()

    def cut_transform(self, us_sim):
        transform_img = cut_get_transform(self.opt_cut, grayscale=False, convert=True, us_sim_flip=False) 
        cut_img_transformed = transform_img(us_sim)
        return cut_img_transformed

    def inference_transformatons(self):
        return cut_get_transform(self.opt_cut, grayscale=False, convert=False, us_sim_flip=False, eval=True) 

    def forward_cut_A(self, us_real):
        no_random_transform = self.inference_transformatons()   #Resize and Normalize -1,1
        data_cut_reconstructed_us = no_random_transform(us_real)
        self.data_cut_real_us['A'] = data_cut_reconstructed_us   # add cut_model.real_B
        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        self.cut_model.forward()


    def forward_cut_B(self, us_sim):
        no_random_transform = self.inference_transformatons()
        data_cut_rendered_us = no_random_transform(us_sim)
        self.data_cut_real_us['B'] = data_cut_rendered_us   # add cut_model.real_B
        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        self.cut_model.forward()

    def get_idtB(self, img):
        self.forward_cut_B(img)
        idt_B = self.cut_model.idt_B
        idt_B = (idt_B / 2 ) + 0.5 # from [-1,1] to [0,1]

        return idt_B

    def print_cut_losses(self, epoch, iters, losses, t_comp, t_data, wandb):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            wandb.log({f"loss_{k}": v, "epoch": epoch})
        print(message) 


    def train_cut(self, module, us_sim_resized, epoch, dataloader_real_us_iterator, iter_data_time):

        try:
            self.data_cut_real_us = next(dataloader_real_us_iterator)
        except StopIteration:
            dataloader_real_us_iterator = iter(self.real_us_train_loader)
            self.data_cut_real_us = next(dataloader_real_us_iterator)

        self.cut_transform = cut_get_transform(self.opt_cut, grayscale=False, convert=False, us_sim_flip=True) 
        self.random_state = torch.get_rng_state()
        self.data_cut_real_us["A"] = self.cut_transform(self.data_cut_real_us["A"]).to(self.opt_cut.device)

        real_us_batch_size = self.data_cut_real_us["A"].size(0)
        self.total_iter += real_us_batch_size

        torch.set_rng_state(self.random_state)
        data_cut_rendered_us = self.cut_transform(us_sim_resized)
        self.data_cut_real_us['B'] = data_cut_rendered_us   # add cut_model.real_B
        
        optimize_start_time = time.time()
        if self.init_cut:    #initialize cut
            print(f"--------------- INIT CUT --------------")
            self.init_cut = False
            self.cut_model.data_dependent_initialize(self.data_cut_real_us)
            self.cut_model.setup(self.opt_cut)               # regular setup: load and print networks; create schedulers
            self.cut_model.parallelize()

        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        # forward
        self.cut_model.forward()

        self.cut_optimize_parameters(module, None)
        self.cut_optimizer_step()

        # calculate loss functions, get gradients, update network weights, backward()
        self.optimize_time = (time.time() - optimize_start_time) / real_us_batch_size * 0.005 + 0.995 * self.optimize_time

        iter_start_time = time.time() 
        if self.total_iter % self.opt_cut.print_freq == 0:
            self.t_data = iter_start_time - iter_data_time

        #Print CUT results
        if self.total_iter % self.opt_cut.display_freq == 0:   # display images and losses on wandb
            self.cut_model.compute_visuals()
            self.cut_plot_figs.append(self.plotter.plot_images(visuals=self.cut_model.get_current_visuals(), epoch=epoch, plot_single=False))
            losses = self.cut_model.get_current_losses()
            self.print_cut_losses(epoch, self.total_iter, losses, self.optimize_time, self.t_data, wandb)

