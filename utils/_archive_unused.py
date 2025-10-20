# Backup of unused defs/classes moved during cleanup.
# --- Moved from Cifar10/Evaluation_Matix.py: FunctionDef get_AUC
def get_AUC(outputs):
	AUC = []
	for i in range(outputs[0].shape[1]):
		fpr, tpr, thresholds = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		AUC.append(metrics.auc(fpr, tpr))
	return np.mean(AUC)


# --- Moved from Cifar10/model_loader.py: FunctionDef weight_ini
def weight_ini(m):
    torch.manual_seed(230)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()


# --- Moved from Cifar10/utils.py: FunctionDef save_dict_to_json
def save_dict_to_json(d, json_path):
	"""Saves dict of floats in json file

	Args:
	d: (dict) of float-castable values (np.float, int, float, etc.)
	json_path: (string) path to json file
	"""
	with open(json_path, 'w') as f:
		# We need to convert the values to float for json (it doesn't accept np.array, np.float, )
		d = {k: float(v) for k, v in d.items()}
		json.dump(d, f, indent=4)
