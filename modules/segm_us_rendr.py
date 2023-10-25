import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from utils.losses import SoftDiceLoss, DiceLoss
from utils.helpers import AddGaussianNoise
from torch.optim.lr_scheduler import StepLR
import os

THRESHOLD = 0.5
SIZE_W = 256
SIZE_H = 256

class SegmentationUSRenderedModule(torch.nn.Module):
    def __init__(self, params, inner_model=None, outer_model=None):
        super(SegmentationUSRenderedModule, self).__init__()
        self.params = params

        self.USRenderingModel = inner_model.to(params.device)

        #define transforms for image and segmentation
        self.rendered_img_masks_random_transf = transforms.Compose([
                transforms.Resize([286, 286], transforms.InterpolationMode.NEAREST),
                transforms.RandomCrop(256),
        ])

        self.rendered_img_random_transf = transforms.Compose([
                transforms.RandomApply([AddGaussianNoise(0., 0.02)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 3))], p=0.5),
        ])

        if params.seg_net_input_augmentations_noise_blur:
            print(self.rendered_img_random_transf)

        if params.seg_net_input_augmentations_rand_crop:
            print(self.rendered_img_masks_random_transf)
        

        if params.outer_model_monai:
            try:
                import monai
            except ImportError:
                print("monai is not installed. Installing...")
                os.system("python -m pip install monai")
                import monai

            channels = (32, 64, 128, 256, 512)
            print('MONAI channels: ', channels, 'DROPOUT: ', params.dropout_ratio)
            self.outer_model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=channels,
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout = params.dropout_ratio
            ).to(params.device)

            self.loss_function = monai.losses.DiceLoss(sigmoid=True)

        else:
            self.outer_model = outer_model.to(params.device)
            self.loss_function = SoftDiceLoss()  # without thresholding is soft DICE loss
            # self.loss_function = DiceLoss()     #default threshold is 0.5


        self.optimizer, self.scheduler = self.configure_optimizers(self.USRenderingModel, self.outer_model)

        print(f'OuterModel On cuda?: ', next(self.outer_model.parameters()).is_cuda)
        print('USRenderingModel On cuda?: ', next(self.USRenderingModel.parameters()).is_cuda)


    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    
    def create_return_dict(self, type, loss, input, file_name, gt_mask, z_hat_pred, us_sim, epoch):
        dict = {f"{type}_images_unet": (input.detach()),
                'file_name': file_name,
                f'{type}_images_gt': gt_mask.detach(),
                f'{type}_images_pred': z_hat_pred.detach(),
                f'{type}_images_us_sim': us_sim.detach(),
                f'{type}_loss': loss.detach(),
                'epoch': epoch}
                
        return dict


    def rendering_forward(self, input):
        us_sim = self.USRenderingModel(input.squeeze()) 
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (SIZE_W, SIZE_H)).float()
        # self.USRenderingModel.plot_fig(us_sim.squeeze(), "us_sim", True)

        return us_sim_resized          
    
    def seg_forward(self, us_sim, label):
        output = self.outer_model.forward(us_sim)

        if self.params.outer_model_monai:
            loss = self.loss_function(output, label)
            pred=output
        else:
            loss, pred = self.loss_function(output, label)

        if self.params.debug: 
            self.USRenderingModel.plot_fig(output.squeeze(), "output_deb", False)
            self.USRenderingModel.plot_fig(label.squeeze(), "label_deb", False)
            
        return loss, pred          


    def step(self, input, label):
        us_sim_resized = self.rendering_forward(input)
        loss, pred = self.seg_forward(self, us_sim_resized, label)

        return loss, us_sim_resized, pred          

    def label_preprocess(self, label):
        label = torch.rot90(label, 3, [2, 3])   
        label = F.resize(label.squeeze(0), (SIZE_W, SIZE_H)).float().unsqueeze(0)
    
        return label

    def get_data(self, batch_data):
        input, label, file_name = batch_data[0].to(self.params.device), batch_data[1].to(self.params.device), batch_data[2]
        
        label = self.label_preprocess(label)

        if self.params.warp_img: 
            label = label.squeeze()
            label = torch.rot90(label, 1)
            label = self.USRenderingModel.warp_img(label)
            label = torch.rot90(label, 3)
            label = label.unsqueeze(0).unsqueeze(0)

        return input, label, file_name


    def validation_step(self, batch_data, epoch, batch_idx=None):
        # label = self.label_preprocess(label)

        input, label, file_name = self.get_data(batch_data)
        loss, us_sim_resized, pred = self.step(input, label)

        dict = self.plot_val_results(input, loss, file_name, label,pred, us_sim_resized, epoch)

        return pred, us_sim_resized, loss, dict


    def plot_val_results(self, input, loss, file_name, label,pred, us_sim_resized, epoch):

        val_images_plot = F.resize(input, (SIZE_W, SIZE_H)).float().unsqueeze(0)
        dict = self.create_return_dict('val', loss, val_images_plot, file_name[0], label, pred, us_sim_resized, epoch)

        return dict


    def configure_optimizers(self, inner_model, outer_model):

        if inner_model is None:
            optimizer = torch.optim.Adam(outer_model.parameters(), self.params.outer_model_learning_rate)
        else:
            optimizer = torch.optim.Adam(
                [
                    {"params": outer_model.parameters(), "lr": self.params.outer_model_learning_rate},
                    {"params": inner_model.parameters(), "lr": self.params.inner_model_learning_rate},
                ],
            )

        if self.params.scheduler: 
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        else:
            scheduler = None

        return optimizer, scheduler
