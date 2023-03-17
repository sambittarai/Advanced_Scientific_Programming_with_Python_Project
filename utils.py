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
