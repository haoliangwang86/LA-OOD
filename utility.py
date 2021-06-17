import os
import csv
import random
import h5py
import numpy as np
from PIL import Image
from joblib import load
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def save_data_hdf5(data, ds_name, file_address, mode="w"):
    """
    r: Readonly, file must exist
    r+: Read/write, file must exist
    w:Create file, truncate if exists
    w- or x:Create file, fail if exists
    a:Read/write if exists, create otherwise (default)
    """
    h5_file = h5py.File(file_address+".hdf5", mode)
    dset = h5_file.create_dataset(ds_name, shape=data.shape, dtype='float64')
    dset[:] = data
    h5_file.close()


def get_dataset_hdf5(ds_name, file_address, without_Ext=True):
    if without_Ext:
        file_address = file_address + ".hdf5"

    h5_file = h5py.File(file_address, "r")
    data = np.copy(h5_file[ds_name])
    h5_file.close()
    return data


def get_dataset(ds_name, is_training_set=False):
    if is_training_set:
        file_address = f'datasets/{ds_name}_train'
    else:
        file_address = f'datasets/{ds_name}'
    data = get_dataset_hdf5('data', file_address)
    return data


class CustomDataset(Dataset):
    def __init__(self, ds_name, is_training=False, transform=None, sample_size=None):
        self.ds_name = ds_name
        self.transform = transform
        self.images = get_dataset(ds_name, is_training)

        if not is_training and sample_size is not None and len(self.images) > sample_size:
            # randomly select OOD samples if there are more samples than required
            idx = random.sample(range(len(self.images)), sample_size)
            self.images = self.images[idx]

        self.size = len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.images[idx]

        img = img.astype(np.uint8)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, 1  # pseudo target


def get_inter_outputs(model_name, ind_name, ds_name, layer_idx, is_training_set=False):
    if is_training_set:
        file_address = f'inter_outputs/{model_name}/{ind_name}_vs_others/{ds_name}_train'
    else:
        file_address = f'inter_outputs/{model_name}/{ind_name}_vs_others/{ds_name}_test'
    data = get_dataset_hdf5(str(layer_idx), file_address)
    return data


def sort_csv_results(filename):
    results = []
    with open(filename, newline='') as csvfile:
        statreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        headers = next(statreader, None)
        for row in statreader:
            results.append([float(x) for x in row])
    for i in range(len(results)):
        results[i][0] = int(results[i][0])
    results = sorted(results, key=lambda row: row[0])
    return headers, results


def get_fpr_at_95_tpr(tpr, fpr):
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def get_statistics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    fpr_at_95_tpr = get_fpr_at_95_tpr(tpr, fpr)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(-labels, -scores)
    aupr_reverse = auc(recall, precision)

    return auroc, fpr_at_95_tpr, aupr, aupr_reverse


def load_ood_detector(model_name, ind_name, layer_idx):
    filename = f"saved_models/{model_name}/{ind_name}_best/{layer_idx}.joblib"
    if os.path.exists(filename):
        model = load(filename)
    else:
        print(f"OOD detector of layer{layer_idx} not found!")
        return
    return model
