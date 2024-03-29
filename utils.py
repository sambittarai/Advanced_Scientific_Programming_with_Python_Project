import os
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import numpy as np
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cc3d
import SimpleITK as sitk
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import nibabel as nib
from medpy import metric

def make_dirs(path, k):
    """
    Create the directories (such as Network_Weights, MIPs, Metrics) necessary for k-fold cross-validation.

    Args:
        path (str): The base path where the cross-validation directories will be created.
        k (int): The number of folds in the cross-validation.

    Returns:
        None
    """
	path_CV = os.path.join(path, "CV_" + str(k))
	if not os.path.exists(path_CV):
		os.mkdir(path_CV)
	path_Network_weights = os.path.join(path_CV, "Network_Weights")
	if not os.path.exists(path_Network_weights):
		os.mkdir(path_Network_weights)
	path_MIPs = os.path.join(path_CV, "MIPs")
	if not os.path.exists(path_MIPs):
		os.mkdir(path_MIPs)
	path_Metrics = os.path.join(path_CV, "Metrics")
	if not os.path.exists(path_Metrics):
		os.mkdir(path_Metrics)

def train(model, train_loader, optimizer, loss_function, device):
    """
    Trains a given model on the given training dataset using the specified optimizer and loss function.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (monai.data.DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimization algorithm to be used during training.
        loss_function (torch.nn.Module): The loss function to be optimized during training.
        device (torch.device): The device on which to perform the training.

    Returns:
        float: The average loss over the entire training epoch.
    """
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
    """
    Extracts the patient ID and scan date from a dictionary of PET-CT files.

    Args:
        files (dict): A dictionary containing the file paths for the PET and CT images.

    Returns:
        tuple: A tuple containing the patient ID and scan date, in the format (patient_id, scan_date).
    """
	pat_scan = files['SUV'].split('PETCT_')[-1]
	pat_id = 'PETCT_' + pat_scan.split('/')[0]
	scan_date = pat_scan.split('/')[1]
	return pat_id, scan_date

def writeTxtLine(input_path, values):
    """
    Appends a new line of comma-separated values to a text file.

    Args:
        input_path (str): The file path of the text file to write to.
        values (list): A list of values to be written to the file.

    Returns:
        None: This function does not return anything.
    """
	with open(input_path, "a") as f:
		f.write("\n")
		f.write("{}".format(values[0]))
		for i in range(1, len(values)):
			f.write(",{}".format(values[i]))

def compute_metrics_validation(GT, pred, pat_ID, scan_date, path):
    """
    Compute metrics such as DICE, True positives, False positives, and False negatives.

    Args:
	GT (numpy array): Contains the ground truth labels for a given patient's scan
	pred (numpy array): Contains the predicted labels for the same scan
	pat_ID (str): The ID of the patient
	scan_date (str): The date of scan
	path (str): Contains the file path to save the computed metrics
    Returns:
        float: The dice score of the given patient scan.
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

	return dice#, fp_freq, fn_freq

def save_model(model, epoch, optimizer, k, path_Output):
    """
    Saves the state of the model, optimizer, and epoch to a .pth.tar file at the specified path.

    Parameters:
    model (torch.nn.Module): PyTorch model to save.
    epoch (int): Current epoch number.
    optimizer (torch.optim.Optimizer): Optimizer used for training.
    k (int): Fold number for cross-validation.
    path_Output (str): Output path to save the model weights.

    Returns:
    None
    """
	best_metric_epoch = epoch + 1
	state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_metric_epoch}
	torch.save(state, os.path.join(path_Output, "CV_" + str(k) + "/Network_Weights/best_model_{}.pth.tar".format(best_metric_epoch)))

def plot_dice(dice, path):
	epoch = [2 * (i + 1) for i in range(len(dice))]
	plt.plot(epoch, dice)
	plt.savefig(path, dpi=400)
	plt.xlabel("Number of Epochs")
	plt.ylabel("DICE")

def validation(args, epoch, optimizer, post_pred, post_label, model, val_loader, device, dice_metric, metric_values, best_metric, k, val_files, path_Output):
	model.eval()
	pat_file = 0
	with torch.no_grad():
		for val_data in tqdm(val_loader):
			pat_id, scan_date = get_patient_id(val_files[pat_file])

			val_inputs, val_labels = (
				val_data["PET_CT"].to(device),
				val_data["SEG"].to(device),
			)
			roi_size = args.roi_size
			sw_batch_size = 4
			val_outputs = sliding_window_inference(
				val_inputs, roi_size, sw_batch_size, model)
			prediction = val_outputs.argmax(dim = 1).data.cpu().numpy()
			prediction = np.squeeze(prediction, axis=0)
			GT = val_labels[0,0,:,:,:].data.cpu().numpy()
			#print("dice: {}".format(DICE_Score(prediction, GT)))

			# compute metric for current iteration
			if len(np.unique(GT)) == 2:
				val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
				val_labels = [post_label(i) for i in decollate_batch(val_labels)]
				dice_metric(y_pred=val_outputs, y=val_labels)

			#Compute DICE, TP, FP, FN, Final Score and save it in a text file under the patient's name.
			path_dice = os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch+1) + ".txt")
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
