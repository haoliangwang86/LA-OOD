import os
import csv
import random
import argparse
import numpy as np
from tqdm import tqdm
from joblib import dump
from itertools import product
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from self_adaptive_shifting import SelfAdaptiveShifting
from torch.multiprocessing import Pool, cpu_count
from global_settings import *
from utility import get_inter_outputs, sort_csv_results


parser = argparse.ArgumentParser()
parser.add_argument('--kernel', default='rbf')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--threshold', type=float, default=0.01)
parser.add_argument('--val_size', default=0.3, type=float,
                    help='Holdout validation set size')
parser.add_argument('--error_rate', default=0.5, type=float,
                    help='Pseudo OOD error rate')
parser.add_argument('--model', default='vgg16', type=str,
                    help='model name')
parser.add_argument('--ind', default='cifar10', type=str,
                    help='InD dataset name')


def main():
    args = parser.parse_args()
    basedir = f"saved_models/{args.model}/{args.ind}_best/"

    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    best_param_filename = basedir + "best_hyperparam.csv"
    fields = ['Layer', 'nu', 'gamma', 'error']
    with open(best_param_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

    fitting_process_filename = basedir + "fitting_process.csv"
    fields = ['Layer', 'nu', 'gamma', 'error', 'error_ind', 'error_ood']
    with open(fitting_process_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

    layers = 0
    if args.model == 'vgg16':
        layers = VGG16_LAYERS
    elif args.model == 'resnet34':
        layers = RESNET34_LAYERS
    elif args.model == 'densenet100':
        layers = DENSENET100_LAYERS

    cpus = cpu_count() - 1  # spare one cpu to avoid locking up the system
    processes = min(cpus, len(layers))

    with Pool(processes=processes) as p:
        p.starmap_async(find_best_ocsvm, product([args.model], [args.ind], layers, [args.error_rate])).get()
        p.close()
        p.join()

    for filename in [best_param_filename, fitting_process_filename]:
        headers, results = sort_csv_results(filename)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerows(results)

    print("All OOD detectors have been trained!")


def find_best_ocsvm(model_name, ds_name, layer_idx, ood_error_rate):
    print(f"layer {layer_idx} OOD detector training started!")
    args = parser.parse_args()
    random.seed(args.seed)

    X = np.array(get_inter_outputs(model_name, ds_name, ds_name, layer_idx, 'train'))

    ss = StandardScaler()
    ss.fit(X)
    X = ss.transform(X)

    train_X, val_X = train_test_split(X, test_size=args.val_size, random_state=args.seed)

    self_adaptive_shifting = SelfAdaptiveShifting(val_X)
    self_adaptive_shifting.edge_pattern_detection(args.threshold)

    pseudo_outlier_X = self_adaptive_shifting.generate_pseudo_outliers()
    pseudo_outlier_Y = -np.ones(len(pseudo_outlier_X))

    val_Y = np.ones(len(val_X))
    print(f"layer {layer_idx} pseudo feature generated.")

    std = np.std(train_X)

    nu_candidates = [0.001]

    if args.model == "densenet":
        gamma_candidates = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    else:
        gamma_candidates = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]

    best_err = 1.0
    best_gamma, best_nu = 1 / (np.size(train_X, -1) * std), 0.5

    basedir = f"saved_models/{model_name}/{ds_name}_best/"

    print(f"layer {layer_idx} fitting started.")
    for nu in nu_candidates:
        for gamma in tqdm(gamma_candidates):
            model = OneClassSVM(gamma=gamma, nu=nu).fit(train_X)
            err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_Y)
            err_t = 1 - np.mean(model.predict(val_X) == val_Y)
            err = ood_error_rate * err_o + (1. - ood_error_rate) * err_t
            if err < best_err:
                best_err = err
                best_gamma = gamma
                best_nu = nu
                print(f"new best - layer {layer_idx}: nu-{nu}, gamma-{gamma}")

                filename = basedir + "fitting_process.csv"
                row = [layer_idx + 1, best_nu, best_gamma, f'{best_err:.4f}', f'{err_t:.4f}', f'{err_o:.4f}']
                with open(filename, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(row)

    filename = basedir + f"{layer_idx}.joblib"
    best_model = OneClassSVM(kernel=args.kernel, gamma=best_gamma, nu=best_nu).fit(X)
    dump(best_model, filename)

    filename = basedir + "best_hyperparam.csv"
    row = [layer_idx + 1, best_nu, best_gamma, f'{best_err:.4f}']
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

    print(f"layer {layer_idx} OOD detector training finished!")


if __name__ == '__main__':
    main()
