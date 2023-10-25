import configargparse

def build_configargparser(parser):
    parser.add_argument("--debug",  action="store_true", help='If True will plot images for local debugging')

    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--project_name", default=None, type=str, help='The wandb project name')

    parser.add_argument("--base_folder_data_path", default="", required=False, type=str, help='Pat')
    parser.add_argument("--base_folder_mask_path", default="", required=False, type=str)
    parser.add_argument("--labelmap_path", default="", required=False, type=str)
    parser.add_argument("--aorta_only",  action="store_true")


    parser.add_argument("--data_dir_real_us_cut_training", default="", required=False, type=str)
    parser.add_argument("--data_dir_real_us_test", default="", required=False, type=str)
    parser.add_argument("--data_dir_real_us_stopp_crit", default="", required=False, type=str)

    parser.add_argument("--output_path", type=str, default="logs")
    parser.add_argument("--device", type=str, required=False, default="cuda")

    #Datatset
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--datamodule", type=str, required=False)
    parser.add_argument("--dataloader_ct_labelmaps", type=str, required=False)
    parser.add_argument("--dataloader_real_us", type=str, required=False)
    parser.add_argument("--dataloader_real_us_test", type=str, required=False)
    parser.add_argument("--n_classes", type=int, required=False)
    parser.add_argument("--pred_label", type=int, required=False)

    # Model
    parser.add_argument("--module", type=str, required=False)
    parser.add_argument("--outer_model_monai",  action="store_true")
    parser.add_argument("--outer_model", type=str, required=False)
    parser.add_argument("--dropout",  action="store_true")
    parser.add_argument("--dropout_ratio", type=float, required=False)
    parser.add_argument("--seg_net_input_augmentations_noise_blur",  action="store_true")
    parser.add_argument("--seg_net_input_augmentations_rand_crop",  action="store_true")

    parser.add_argument("--inner_model", type=str, required=False)
    parser.add_argument("--warp_img",  action="store_true")

    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--batch_size_manual", type=int, required=False)
    parser.add_argument("--global_learning_rate", type=float, required=False)

    parser.add_argument("--discr_model_learning_rate", type=float, required=False)
    parser.add_argument("--inner_model_learning_rate", type=float, required=False)
    parser.add_argument("--outer_model_learning_rate", type=float, required=False)
    parser.add_argument("--scheduler",  action="store_true")
    parser.add_argument("--grad_clipping",  action="store_true")
    parser.add_argument("--lambda_G_loss", type=float, required=False)

    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--min_epochs", default=10, type=int)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--validate_every_n_steps", default=1, type=int)
    
    parser.add_argument("--stopping_crit_gt_imgs", action="store_true")
    parser.add_argument("--epochs_check_stopp_crit", default=0, type=int)
    parser.add_argument("--nr_imgs_to_plot", default=0, type=int)

    parser.add_argument("--use_idtB", action="store_true")
    
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--log_default_renderer", action="store_true")

    parser.add_argument("--wandb_conf", action="store_true")

    known_args = parser.parse_args()
    
    return parser, known_args