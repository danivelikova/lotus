import matplotlib.pyplot as plt
from matplotlib import gridspec
import torchvision
import torch
import wandb
import io
from PIL import Image, ImageFont, ImageDraw
import matplotlib.font_manager as fm
import numpy as np
from operator import *

N = 16
class Plotter():
    def __init__(self):
        self.figs = []
        self.plot_figs = []
        self.train_fold_dict = []
        self.val_fold_dict = []
        self.val_fold_dict_epoch = []
        self.test_fold_dict_epoch = []
        self.plt_test_img_cnt = 0

    def log_image(self, plt, caption):
        print('log image: ', caption)
        wandb.log({caption: [wandb.Image(plt, caption=caption)]}, commit=True)

    def plot(self, imgs):
        rows = 1
        cols = len(imgs)
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        spec = gridspec.GridSpec(rows, cols, fig, wspace=0, hspace=0)
        spec.tight_layout(fig)
        for idx, img in enumerate(imgs):
            img = img.cpu().detach()
            if idx==2:  # prediction blended over the us_sim image
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none', norm=None)
                us_sim_img = imgs[1].cpu().detach()
                plt.subplot(spec[0, idx]).imshow(us_sim_img, cmap='gray', alpha=0.4, vmin=0, vmax=1, interpolation='none', norm=None)
            elif idx==0:    # original ct_slice
                img = np.rot90(img, 3)
                plt.subplot(spec[0, idx]).imshow(img)
            elif idx==1:    #us_sim
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none', norm=None)
            else:   #gt_mask
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none', norm=None)
                us_sim_img = imgs[1].cpu().detach()
                plt.subplot(spec[0, idx]).imshow(us_sim_img, cmap='gray', alpha=0.4, vmin=0, vmax=1, interpolation='none', norm=None)
            plt.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("/fig.png", bbox_inches='tight')
        # plt.savefig('results_test/fig_test.png')
        # plt.show()

        return fig

    # Converts a Tensor array into a numpy image array
    def tensor2im(self, input_image, label, imtype=np.uint8):

        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array

            if 'gt_label' in label:
                image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0   # post-processing: tranpose and scaling
            else:
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)


    def plot_images(self, visuals, epoch, plot_single):
        labels = ""
        images_pil = []
        label_first = True
        for label, image in visuals.items():
            
            image = self.tensor2im(image, label)
            pil_fig = torchvision.transforms.ToPILImage()(image)#.convert('L')

            if label_first:
                plot_fig_pil_draw = ImageDraw.Draw(pil_fig)
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), 15)
                curr_epoch = 'epoch=' + str(epoch)
                plot_fig_pil_draw.text((2, 2), curr_epoch, (255), font=font)
                label_first = False

            pil_fig_t = torchvision.transforms.ToTensor()(pil_fig)
            images_pil.append(pil_fig_t)
            labels += label + '|'

        # print('LABELS: ', labels)

        if plot_single: wandb.log({str(labels): [wandb.Image(image) for image in images_pil]})

        return torchvision.utils.make_grid(images_pil)


    def add_text_to_plot(self, plot_fig, text, epoch):
        if text:
            text = 'epoch=' + str(epoch) + '_' + text
        fontsize = 28

        buf = io.BytesIO()
        plot_fig.savefig(buf, format='png')
        buf.seek(0)
        plot_fig_pil = Image.open(buf)
        plot_fig_pil_draw = ImageDraw.Draw(plot_fig_pil)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), fontsize)
        plot_fig_pil_draw.text((10, 10), text, (255, 255, 255), font=font)
        plot_fig_pil_t = torchvision.transforms.ToTensor()(plot_fig_pil)

        plot_fig.clf()  #clear the figure
        plt.close(plot_fig)     #closes windows associated with the fig
        return plot_fig_pil_t


    def plot_stopp_crit(self, caption, imgs, img_text, epoch, plot_single):
        rows = 1
        cols = len(imgs)
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        spec = gridspec.GridSpec(rows, cols, fig, wspace=0, hspace=0)
        spec.tight_layout(fig)
        for idx, img in enumerate(imgs):
            img = img.cpu().detach().squeeze()
            if 'int' in str(img.dtype) :
                img = np.rot90(img, 3)
                plt.subplot(spec[0, idx]).imshow(img)
            else:
                plt.subplot(spec[0, idx]).imshow(img[:, :], cmap='gray', vmin=0, vmax=1, interpolation='none', norm=None)
            plt.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        
        plot_fig_pil_t = self.add_text_to_plot(fig, text=img_text, epoch=epoch)
        if plot_single: self.log_image(plot_fig_pil_t, caption)

        return plot_fig_pil_t
    

    def validation_batch_end(self, outputs):
        self.val_fold_dict_epoch.append(outputs)


    def validation_epoch_end(self):
        sorted_best_loss_val_dic = sorted(self.val_fold_dict_epoch, key=lambda d: d['file_name']) 

        get_top_last = [sorted_best_loss_val_dic[:N], sorted_best_loss_val_dic[-N:]]
        caption = [f"best_{N}", f"worst_{N}"]

        for i in range(2):
            for x in get_top_last[i]:
                plot_fig = self.plot([x["val_images_unet"].squeeze(), x["val_images_us_sim"].squeeze(), x["val_images_pred"].squeeze(), x["val_images_gt"].squeeze()])
                plot_fig_pil_t = self.add_text_to_plot(plot_fig, "volume_" + x["file_name"][0], x["epoch"])
                self.plot_figs.append(plot_fig_pil_t)

            self.log_image(torchvision.utils.make_grid(self.plot_figs), caption[i] +
                           "val |orig_input|us_sim|pred|gt_mask")

            del self.plot_figs[:]
    
        del self.val_fold_dict_epoch[:]


    def log_us_rendering_values(self, inner_model, step):
        
        us_rendering_values = [inner_model.acoustic_impedance_dict, inner_model.attenuation_dict, 
                                inner_model.mu_0_dict, inner_model.mu_1_dict, inner_model.sigma_0_dict]

        us_rendering_maps = ['acoustic_imp_', 'atten_', 'mu_0_', 'mu_1_', 'sigma_0_' ]
        for map, dict in zip(us_rendering_maps, us_rendering_values):
            for label, value in zip(inner_model.labels, dict):
                wandb.log({map+label: value}, commit=False)
            wandb.log({"step":step})


# def log_model_gradients_tb(model, step):
#     for tag, value in model.named_parameters():
#         if value.grad is not None:
#             tb_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)
