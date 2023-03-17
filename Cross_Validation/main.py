import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from tqdm import tqdm
import pandas as pd
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord
)
from monai.networks.layers import Norm
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric, compute_meandice
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
import warnings
warnings.filterwarnings("ignore")
#torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch_dataloader import prepare_data
import sys
sys.path.insert(0, '/media/sambit/Seagate Expansion Drive/G/Uppsala University/Courses/Semesters/2023/1_January_July/Advanced Scientific Programming with Python/Project/Advanced_Scientific_Programming_with_Python_Project')
from config import parse_args
from network_architecture import build_network
from utils import make_dirs, train, validation, plot_dice

def main(args):
	#Path Initialization
	path_df_filtered = args.path_df_patients_with_tumors
	df_filtered = pd.read_csv(path_df_filtered)
	path_Output = args.path_CV_Output

	#Cross validation parameters initialization
	K = args.K_fold_CV
	max_epochs = args.max_epochs
	val_interval = args.validation_interval
	best_metric = args.best_metric

	for k in range(K):
		print("Cross Valdation for fold: {}".format(k))
		#GPU configuration
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		#Building the network architecture
		model = build_network(args, device)
		#Optimization
		loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
		dice_metric = DiceMetric(include_background=False, reduction="mean")

		#Create all the relevant directories
		make_dirs(path_Output, k)
		#Dataloader Preparation
		if k == (K - 1):
			df_val = df_filtered[100*k:].reset_index(drop=True)
		else:
			df_val = df_filtered[100*k:100*k+100].reset_index(drop=True)
		df_train = df_filtered[~df_filtered.scan_date.isin(df_val.scan_date)].reset_index(drop=True)
		train_loader, val_loader, val_files = prepare_data(df_train, df_val)

		post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
		post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

		for epoch in tqdm(range(max_epochs)):
			#Training
			epoch_loss = train(model, train_loader, optimizer, loss_function, device)
			lr_scheduler.step()
			print(f"Training epoch {epoch + 1} average loss: {epoch_loss:.4f}")
			#Validation
			if (epoch + 1) % val_interval == 0:
				metric_values, best_metric_new = validation(args, epoch, optimizer, post_pred, post_label, model, val_loader, device, dice_metric, metric_values, best_metric, k, val_files, path_Output)
				best_metric = best_metric_new
				print("Validation DICE for epoch_{} : {}".format(epoch, best_metric_new))

			#Save and plot DICE
			np.save(os.path.join(path_Output, "CV_" + str(k) + "/DICE.npy"), metric_values)
			path_dice = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_dice.jpg")
			if len(metric_values) > 2:
				plot_dice(metric_values, path_dice)	

if __name__ == "__main__":
	args = parse_args()
	main(args)