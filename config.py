def parse_args():
    parser = argparse.ArgumentParser()

    #################################Common

    #Data Path
    parser.add_argument("--data_path", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Data/FDG-PET-CT-Lesions", help="This path contains medical scans and their corresponding segmentation masks for all the patients.")

    #Filtered DataFrame
    parser.add_argument("--path_df", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df.csv", help="DataFrame containing paths (CT, SUV, SEG) for all the patients.")
    parser.add_argument("--path_df_patients_with_tumors", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df_patients_with_tumors.csv", help="DataFrame containing paths (CT, SUV, SEG) for all the patients with tumor.")
    parser.add_argument("--path_df_patients_without_tumors", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df_patients_without_tumors.csv", help="DataFrame containing paths (CT, SUV, SEG) for all the patients without tumor.")

    #Best Network Weights
    parser.add_argument("--path_checkpoint_CV_0", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Cross_Validation/Output/UNET_3D/5_Fold_CV_patch_160/CV_0/Network_Weights/best_model_92.pth.tar", help="Best Model for CV_0.")
    parser.add_argument("--path_checkpoint_CV_1", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Cross_Validation/Output/UNET_3D/5_Fold_CV_patch_160/CV_1/Network_Weights/best_model_116.pth.tar", help="Best Model for CV_1.")
    parser.add_argument("--path_checkpoint_CV_2", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Cross_Validation/Output/UNET_3D/5_Fold_CV_patch_160/CV_2/Network_Weights/best_model_82.pth.tar", help="Best Model for CV_2.")
    parser.add_argument("--path_checkpoint_CV_3", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Cross_Validation/Output/UNET_3D/5_Fold_CV_patch_160/CV_3/Network_Weights/best_model_116.pth.tar", help="Best Model for CV_3.")
    parser.add_argument("--path_checkpoint_CV_4", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Cross_Validation/Output/UNET_3D/5_Fold_CV_patch_160/CV_4/Network_Weights/best_model_144.pth.tar", help="Best Model for CV_4.")

    ##################################Cross_Validation

    #Dataloader
    parser.add_argument("--batch_size_train", default=2, help="Batch Size for Train_Loader.")
    parser.add_argument("--batch_size_val", default=1, help="Batch Size for Validation_Loader.")
    parser.add_argument("--num_workers", default=2, help="Number of Workers.")
    parser.add_argument("--CT_min", default=-100, help="Minimum CT intensity value during intensity clipping.")
    parser.add_argument("--CT_max", default=250, help="Maximum CT intensity value during intensity clipping.")
    parser.add_argument("--SUV_min", default=0, help="Minimum SUV intensity value during intensity clipping.")
    parser.add_argument("--SUV_max", default=15, help="Maximum SUV intensity value during intensity clipping.")
    parser.add_argument("--pos", default=4, help="Data Sampling (Pos to Neg) Ratio for tumor class.")
    parser.add_argument("--neg", default=1, help="Data Sampling (Pos to Neg) Ratio for background class.")
    parser.add_argument("--num_samples", default=2, help="Number of 3D Samples extracted from a single patient during Training.")

    #Network Architecture
    parser.add_argument("--dimensions", default=3, help="Dimension of the UNET.")
    parser.add_argument("--in_channels", default=2, help="Number of input channels.")
    parser.add_argument("--out_channels", default=2, help="Number of segmentation classes.")
    parser.add_argument("--kernel_channels", default=(16, 32, 64, 128, 256), help="Sequence of kernels.")
    parser.add_argument("--strides", default=(2, 2, 2, 2), help="Sequence of strides.")
    parser.add_argument("--residual_units", default=2, help="Number of residual units in the UNET.")
    parser.add_argument("--dropout", default=0.20, help="Dropout Percentage.")
    
    #Optimizer
    parser.add_argument("--lr", default=1e-4, help="Learning Rate.")
    parser.add_argument("--weight_decay", default=1e-5, help="Weight Decay.")
    parser.add_argument("--momentum", default=0.99, help="Momentum used in SGD.")

    #Training Parameters
    parser.add_argument("--max_epochs", default=1000, help="Maximum Number of Epochs for Training.")
    parser.add_argument("--validation_interval", default=1, help="Number of interval after which we want to perform validation.")
    parser.add_argument("--best_metric", default=0, help="Best Dice Metric.")
    parser.add_argument("--best_metric_epoch", default=-1, help="Epoch corresponding to Best Dice Metric.")
    parser.add_argument("--roi_size", default=(160,160,160), help="Image Patch used for training.")
    parser.add_argument("--K_fold_CV", default=5, help="K fold Cross Validation.")
    parser.add_argument("--pre_trained_weights", default=False, help="If True, then load the pretrained weights to the network and resume training.")

    #Save Paths
    parser.add_argument("--path_CV_Output", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Cross_Validation/Output/UNET_3D/5_Fold_CV_160_Melanoma", help="Output Path.")

    ##################################Inference
    #parser.add_argument("--roi_size_val", default=(160,160,160), help="Image patches during validation.")
    #parser.add_argument("--TP_threshold", default=0.01, help="Threshold for DICE score used during tumor detection.")
    #parser.add_argument("--spacing", default=(2.0364201068878174, 2.0364201068878174, 3.0), help="Threshold for DICE score used during tumor detection.")
    #parser.add_argument("--path_Inference_Output", default="/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Inference/Output/Final_New", help="Inference Path.")

    args = parser.parse_args()
    return args