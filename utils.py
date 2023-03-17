import os
from tqdm import tqdm

def make_dirs(path, k):
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
