import numpy as np
import torch
import os
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
#from create_dataset import prepare_data

#import sys
#sys.path.insert(0, '/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET')
#from config import parse_args
#from utils import DICE_Score, train, validation, get_df, filter_tumor_scans, make_dirs, plot_dice, load_checkpoint
import warnings
warnings.filterwarnings("ignore")
#torch.multiprocessing.set_sharing_strategy('file_system')

def prepare_data(df_train, df_val):
	#df_train = get_df(args.data_path_train)
	CT_train = sorted(df_train['CT'].tolist())
	SUV_train = sorted(df_train['SUV'].tolist())
	SEG_train = sorted(df_train['SEG'].tolist())

	#df_val = get_df(args.data_path_val)
	CT_val = sorted(df_val['CT'].tolist())
	SUV_val = sorted(df_val['SUV'].tolist())
	SEG_val = sorted(df_val['SEG'].tolist())

	train_files = [
		{"SUV": SUV_name, "CT": CT_name, "SEG": SEG_name}
		for SUV_name, CT_name, SEG_name in zip(SUV_train, CT_train, SEG_train)
	]
	val_files = [
		{"SUV": SUV_name, "CT": CT_name, "SEG": SEG_name}
		for SUV_name, CT_name, SEG_name in zip(SUV_val, CT_val, SEG_val)
	]

	# define transforms for image and segmentation
	train_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV", "CT", "SEG"]),
	        AddChanneld(keys=["SUV", "CT", "SEG"]),
	        #EnsureChannelFirstd(keys=["SUV", "CT", "SEG"]),
	        ScaleIntensityRanged(
	            keys=["CT"], a_min=-100, a_max=250,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        ScaleIntensityRanged(
	            keys=["SUV"], a_min=0, a_max=15,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        CropForegroundd(keys=["SUV", "CT", "SEG"], source_key="CT"),
	        RandCropByPosNegLabeld(
	            keys=["SUV", "CT", "SEG"],
	            label_key="SEG",
	            spatial_size=(160, 160, 160),
	            pos=4,
	            neg=1,
	            num_samples=4,
	            #image_key="image",
	            image_threshold=0,
	        ),
	        #DivisiblePadd(keys=["SUV", "CT", "SEG"], k=64, allow_missing_keys=False),

	        #Augmentations
	        #RandFlipd(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axis=0),
	        #RandFlipd(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axis=1),
	        #RandFlipd(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axis=2),
	        #RandAffined(keys=['SUV', "CT", 'SEG'], prob=0.3, translate_range=10),
	        #RandRotate90d(keys=["SUV", "CT", "SEG"], prob=0.4, spatial_axes=[0, 2]),
	        #RandGaussianNoised(keys=["SUV", "CT"], prob=0.2),
	        #RandShiftIntensityd(keys=["SUV"], offsets=(0,30), prob=0.2),

	        ConcatItemsd(keys=["SUV", "CT"], name="PET_CT", dim=0),# concatenate pet and ct channels

	        #EnsureTyped(keys=["image", "label"]),
	        #ToTensord(keys=["SUV", "SEG"]),
	        ToTensord(keys=["PET_CT", "SEG"]),
	    ]
	)

	val_transforms = Compose(
	    [
	        LoadImaged(keys=["SUV", "CT", "SEG"]),
	        AddChanneld(keys=["SUV", "CT", "SEG"]),
	        #EnsureChannelFirstd(keys=["SUV", "CT", "SEG"]),
	        ScaleIntensityRanged(
	            keys=["CT"], a_min=-100, a_max=250,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        ScaleIntensityRanged(
	            keys=["SUV"], a_min=0, a_max=15,
	            b_min=0.0, b_max=1.0, clip=True,
	        ),
	        CropForegroundd(keys=["SUV", "CT", "SEG"], source_key="CT"),
	        #RandCropByPosNegLabeld(
	        #    keys=["SUV", "CT", "SEG"],
	        #    label_key="SEG",
	        #    spatial_size=(160, 160, 160),
	        #    pos=4,
	        #    neg=1,
	        #    num_samples=2,
	        #    #image_key="image",
	        #    image_threshold=0,
	        #),
	        ConcatItemsd(keys=["SUV", "CT"], name="PET_CT", dim=0),# concatenate pet and ct channels

	        #EnsureTyped(keys=["PET_CT", "SEG"]),
	        #ToTensord(keys=["SUV", "SEG"]),
	        ToTensord(keys=["PET_CT", "SEG"]),
	    ]
	)

	train_dset = Dataset(data=train_files, transform=train_transforms)
	train_loader = DataLoader(train_dset, batch_size=1, num_workers=4, collate_fn = list_data_collate)

	val_dset = Dataset(data=val_files, transform=val_transforms)
	val_loader = DataLoader(val_dset, batch_size=1, num_workers=4, collate_fn = list_data_collate)
	return train_loader, val_loader, val_files

def train(model, train_loader, optimizer, loss_function, device):
	model.train()
	epoch_loss = 0
	step = 0
	for train_data in tqdm(train_loader):
		step += 1
		inputs, labels = (
			train_data["PET_CT"].to(device),
			train_data["SEG"].to(device),
		)
		#print(inputs.shape)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
	epoch_loss /= step

	return epoch_loss

def get_patient_id(files):
	pat_scan = files['SUV'].split('PETCT_')[-1]
	pat_id = 'PETCT_' + pat_scan.split('/')[0]
	scan_date = pat_scan.split('/')[1]
	return pat_id, scan_date

def writeTxtLine(input_path, values):
	with open(input_path, "a") as f:
		f.write("\n")
		f.write("{}".format(values[0]))
		for i in range(1, len(values)):
			f.write(",{}".format(values[i]))

def compute_metrics_validation(GT, pred, pat_ID, scan_date, path):
	"""
	Computes, DICE, TP, FP, FN.
	Also Computes the Final Score, i.e. (0.5*DICE + 0.25*FP + 0.25*FN)
	"""
	pred = np.where(pred>1, 1, pred)

	if len(np.unique(GT)) == 1:
		dice = 11
		TP = 0
		disease_type = "Normal_Scan"
	else:
		dice = np.sum(pred[GT==1])*2.0 / (np.sum(pred) + np.sum(GT))
		TP = np.where(GT != pred, 0, GT)
		disease_type = "Tumorous_Scan"

	#TP = np.where(GT != pred, 0, GT)
	FP = pred - GT
	FP = np.where(FP == -1, 0, FP)
	FN = GT - pred
	FN = np.where(FN == -1, 0, FN)

	if np.count_nonzero(GT == 1) == 0:
		denominator = 1
	else:
		denominator = np.count_nonzero(GT == 0)

	tp_freq = np.count_nonzero(TP == 1)
	tp_percent = tp_freq/denominator
	fp_freq = np.count_nonzero(FP == 1)
	fp_percent = fp_freq/denominator
	fn_freq = np.count_nonzero(FN == 1)
	fn_percent = fn_freq/denominator

	if not os.path.isfile(path):
		with open(path, "w") as f:
			f.write("ID,scan_date,DISEASE_TYPE,DICE,TP,TP_%,FP,FP_%,FN,FN_%")
	writeTxtLine(path, [pat_ID,scan_date,disease_type,dice,tp_freq,tp_percent,fp_freq,fp_percent,fn_freq,fn_percent])

	return dice, fp_freq, fn_freq

def save_model(model, epoch, optimizer, k, path_Output):
	best_metric_epoch = epoch + 1
	state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_metric_epoch}
	torch.save(state, os.path.join(path_Output, "CV_" + str(k) + "/Network_Weights/best_model_{}.pth.tar".format(best_metric_epoch)))


def validation(epoch, optimizer, post_pred, post_label, model, val_loader, device, dice_metric, metric_values, best_metric, k, val_files, path_Output):
	model.eval()
	pat_file = 0
	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date = get_patient_id(val_files[pat_file])

			val_inputs, val_labels = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
			)
			roi_size = (160,160,160)
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:,:].data.cpu().numpy()
			print("dice: {}".format(DICE_Score(prediction, GT)))

			# compute metric for current iteration
			if len(np.unique(GT)) == 2:
				val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
				val_labels = [post_label(i) for i in decollate_batch(val_labels)]
				dice_metric(y_pred=val_outputs, y=val_labels)
			#del val_data

			#Compute DICE, TP, FP, FN, Final Score and save it in a text file under the patient's name.
			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch+1) + ".txt")
			#print("unique: ", np.unique(GT))
			#print("len(np.unique(GT))", len(np.unique(GT)))
			dice, fp, fn = compute_metrics_validation(GT, prediction, pat_id, scan_date, path_dice)

			#Generate MIPs for the predictions when the current metric (DICE) is > best_metric (DICE).
			SUV = val_inputs.data.cpu().numpy()[0,0,:,:,:]
			path_MIP = os.path.join(path_Output, "CV_" + str(k), "MIPs", str(pat_id) + "_" + str(scan_date) + ".jpg")
			#overlay_segmentation(SUV, prediction, GT, path_MIP)
			pat_file += 1

		#aggregate the final mean dice result
		metric = dice_metric.aggregate().item()
		#reset the status for next validation round
		dice_metric.reset()
		print("Validation DICE: {}".format(metric))
		metric_values.append(metric)
		#Save the model if DICE is increasing
		if metric > best_metric:
			best_metric = metric
			save_model(model, epoch, optimizer, k, path_Output)

	return metric_values, best_metric

def plot_dice(dice, path):
	epoch = [2 * (i + 1) for i in range(len(dice))]
	plt.plot(epoch, dice)
	plt.savefig(path, dpi=400)
	plt.xlabel("Number of Epochs")
	plt.ylabel("DICE")

def main():
	#K = args.K_fold_CV
	K = 5
	#path_df_filtered = args.path_df_patients_with_tumors
	path_df_filtered = "/media/sambit/HDD/Sambit/Projects/U-CAN/autoPET_2022/Tumor_Detection/3D_UNET/Save_Path/df_patients_with_tumors.csv"
	df_filtered = pd.read_csv(path_df_filtered)
	#path_Output = args.path_CV_Output
	path_Output = "/media/sambit/Seagate Expansion Drive/G/Uppsala University/Courses/Semesters/2023/1_January_July/Advanced Scientific Programming with Python/Project/Advanced_Scientific_Programming_with_Python_Project/Output"

	for k in tqdm(range(K)):
		#print("Cross Valdation for fold: {}".format(k))
		max_epochs = 1000
		#val_interval = args.validation_interval
		val_interval = 1
		#best_metric = args.best_metric
		best_metric = 0
		#best_metric_epoch = args.best_metric_epoch
		print("Network Initialization")
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = UNet(
					dimensions=3,
					in_channels=2,
					out_channels=2,
					channels=(16, 32, 64, 128, 256),
					strides=(2, 2, 2, 2),
					num_res_units=2,
					norm=Norm.BATCH,
					dropout=0,
				).to(device)
		#model = torch.nn.DataParallel(model).to(device)
		

		loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
		dice_metric = DiceMetric(include_background=False, reduction="mean")

		#Make all the relevant directories
		make_dirs(path_Output, k)
		#Dataloader Preparation
		if k == (5 - 1):
			df_val = df_filtered[100*k:].reset_index(drop=True)
		else:
			df_val = df_filtered[100*k:100*k+100].reset_index(drop=True)
		df_train = df_filtered[~df_filtered.scan_date.isin(df_val.scan_date)].reset_index(drop=True)

		train_loader, val_loader, val_files = prepare_data(df_train, df_val)
		print("Length of Train Loader: {} & Validation Loader: {}".format(len(train_loader), len(val_loader)))
		#train_loader, val_loader, val_files = prepare_data(args, df_train_new, df_val_new)

		post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
		post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

		
		for epoch in range(max_epochs):
			#epoch += 7
			#Training
			epoch_loss = train(model, train_loader, optimizer, loss_function, device)
			lr_scheduler.step()
			#epoch_loss_values.append(epoch_loss)
			print(f"Training epoch {epoch + 1} average loss: {epoch_loss:.4f}")
			#Validation
			if (epoch + 1) % val_interval == 0:
				metric_values, best_metric_new = validation(epoch, optimizer, post_pred, post_label, model, val_loader, device, dice_metric, metric_values, best_metric, k, val_files, path_Output)
				best_metric = best_metric_new

			#Save and plot DICE
			np.save(os.path.join(path_Output, "CV_" + str(k) + "/DICE.npy"), metric_values)
			path_dice = os.path.join(path_Output, "CV_" + str(k), "epoch_vs_dice.jpg")
			if len(metric_values) > 2:
				plot_dice(metric_values, path_dice)
			

if __name__ == "__main__":
	main()
	print("Done")