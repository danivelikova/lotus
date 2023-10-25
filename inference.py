import torch
from tqdm.auto import tqdm
import torchvision
import configargparse
from utils.plotter import Plotter
from utils.configargparse_arguments import build_configargparser
from utils.utils import argparse_summary
from cut.lotus_options import LOTUSOptions
import helpers
import trainer

if __name__ == "__main__":

    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_cut = LOTUSOptions().parse()   # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

    argparse_summary(hparams, parser)

    real_us_gt_test_dataloader, _ = helpers.load_real_us_gt_test_data(hparams)
    plotter = Plotter()    
    trainer = trainer.Trainer(hparams, opt_cut, plotter) 
    
    seg_network_ckpt = './checkpoints/best_checkpoint_seg_renderer_test_loss_XXXXXX.pt'     #replace with your ckpt after training
    trainer.module.load_state_dict(torch.load())
    
    cut_network_ckpt = './checkpoints/best_checkpoint_CUT_XXXXX.pt'     #replace with your ckpt after training
    checkpoint = torch.load()
    # Create a new dictionary with keys without the "module." prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    trainer.cut_trainer.cut_model.netG.load_state_dict(new_state_dict)

    
    # ------------------------------------------------------------------------------------------------------------------------------
    #             INFER REAL US IMGS THROUGH THE CUT+SEG NET - INFER THE WHOLE TEST SET
    # ------------------------------------------------------------------------------------------------------------------------------
    gt_test_imgs_plot_figs = []
    testset_losses = []
    hausdorff_epoch = []

    with torch.no_grad():
        for nr, batch_data_real_us_test in tqdm(enumerate(real_us_gt_test_dataloader), total=len(real_us_gt_test_dataloader), ncols= 100, position=0, leave=True):
            
            real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to(hparams.device), batch_data_real_us_test[1].to(hparams.device).float()
            reconstructed_us_testset = trainer.cut_trainer.cut_model.netG(real_us_test_img)

            reconstructed_us_testset = (reconstructed_us_testset / 2 ) + 0.5 # from [-1,1] to [0,1]

            testset_loss, seg_pred  = trainer.module.seg_forward(reconstructed_us_testset, real_us_test_img_label)
            print(f'testset_loss: {testset_loss}')
            testset_losses.append(testset_loss)

            if hparams.logging and nr < hparams.nr_imgs_to_plot:
                real_us_test_img = (real_us_test_img / 2 ) + 0.5 # from [-1,1] to [0,1]

                plot_fig_gt = plotter.plot_stopp_crit(caption="testset_gt_|real_us|reconstructed_us|seg_pred|gt_label",
                                        imgs=[real_us_test_img, reconstructed_us_testset, seg_pred, real_us_test_img_label], 
                                        img_text='loss=' + "{:.4f}".format(testset_loss.item()), epoch='', plot_single=False)
                gt_test_imgs_plot_figs.append(plot_fig_gt)

    if len(gt_test_imgs_plot_figs) > 0: 
        image_grid = torchvision.utils.make_grid(gt_test_imgs_plot_figs)

        from PIL import Image
        image_np = image_grid.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype('uint8'))

        # Save the image locally
        image_pil.save('results_test/gt_test_imgs_plot_figs.png')


    avg_testset_loss = torch.mean(torch.stack(testset_losses))
    std_testset_loss = torch.std(torch.stack(testset_losses))
    print(f'testset_gt_loss_epoch: {avg_testset_loss}')
    print(f'testset_gt_loss_std_epoch: {std_testset_loss}')
