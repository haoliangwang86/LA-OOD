import os
import csv
import argparse
import numpy as np
from itertools import product
from torch.multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from global_settings import *
from utility import get_inter_outputs, get_statistics, load_ood_detector, sort_csv_results


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vgg16', type=str,
                    help='model name')
parser.add_argument('--ind', default='cifar10', type=str,
                    help='InD dataset name')


def detect_oods_for_all_ood_datasets():
    args = parser.parse_args()

    results = []
    for ood_name in OOD_LIST:
        result = detect_oods(args.model, args.ind, ood_name)
        results.append(result)

    print(f"LA-OOD results for {args.model} backbone, {args.ind} InD:")
    print(f"{'OOD': <15}{'FPR-at-95%-TPR' : ^15}{'AUROC' : ^15}{'AUPR-out' : ^15}{'AUPR-in' : >15}")
    for i in range(len(OOD_LIST)):
        print(f"{OOD_LIST[i]: <15}{100*results[i][0]:^15.2f}{100*results[i][1]:^15.2f}"
              f"{100*results[i][2]:^15.2f}{100*results[i][3]:>15.2f}")


def detect_oods(model_name, ind_name, ood_name):
    print(f"Detecting OODs for {model_name}, {ind_name} vs. {ood_name}:")
    print(f"Layer idx\tAUROC")

    if not os.path.exists("results"):
        os.makedirs("results")
    result_filename = f"results/{model_name}-{ind_name}-vs-{ood_name}.csv"
    fields = ['Layer', 'FPR-at-95%-TPR', 'AUROC', 'AUPR-out', 'AUPR-in',
              'InD-correct', 'OOD-correct', 'Max-score-counts(ID/OOD)']

    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

    scores, ind_length, ood_length = compute_ood_scores_of_all_layers(model_name, ind_name, ood_name, result_filename)

    max_scores = scores.max(axis=0)

    y_test = [-1] * ind_length + [1] * ood_length  # OOD as positive, ID as negative

    auroc, fpr_at_95_tpr, aupr_out, aupr_in = get_statistics(y_test, max_scores)

    headers, results = sort_csv_results(result_filename)

    # Append the number of samples received their max scores at each layer
    max_score_ind_idx = np.argmax(scores[:, :ind_length], axis=0)
    max_score_ind_unique, max_score_ind_counts = np.unique(max_score_ind_idx, return_counts=True)
    max_score_ood_idx = np.argmax(scores[:, ind_length:], axis=0)
    max_score_ood_unique, max_score_ood_counts = np.unique(max_score_ood_idx, return_counts=True)
    for i in range(len(results)):
        ind_counts = 0
        ood_counts = 0
        if i in max_score_ind_unique:
            ind_counts = max_score_ind_counts[np.where(max_score_ind_unique == i)[0][0]]
        if i in max_score_ood_unique:
            ood_counts = max_score_ood_counts[np.where(max_score_ood_unique == i)[0][0]]
        results[i].append(f"{ind_counts}/{ood_counts}")

    best_idx = np.argmax(results, axis=0)[2]    # best layer
    best = results[best_idx]

    la_ood = ["LA-OOD", f'{100*fpr_at_95_tpr:.2f}', f'{100*auroc:.2f}', f'{100*aupr_out:.2f}', f'{100*aupr_in:.2f}']
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(results)
        csvwriter.writerow(["Best layer:"])
        csvwriter.writerow(best)
        csvwriter.writerow(la_ood)

    print(f"LA-OOD:\t\t{100*auroc:.2f}")
    print()
    return auroc, fpr_at_95_tpr, aupr_out, aupr_in


def compute_ood_scores_of_all_layers(model_name, ind_name, ood_name, result_filename):
    layers = 0
    if model_name == 'vgg16':
        layers = VGG16_LAYERS
    elif model_name == 'resnet34':
        layers = RESNET34_LAYERS
    elif model_name == 'densenet100':
        layers = DENSENET100_LAYERS

    cpus = cpu_count() - 1  # spare one cpu to avoid locking up the system
    processes = min(cpus, len(layers))

    with Pool(processes=processes) as p:
        outputs = p.starmap_async(
            compute_ood_scores, product([model_name], [ind_name], [ood_name], layers, [result_filename])).get()
        p.close()
        p.join()

    results = []
    for output in outputs:
        results.append(output[0])
    return np.array(results), outputs[0][1], outputs[0][2]


def compute_ood_scores(model_name, ind_name, ood_name, layer_idx, result_filename):
    ood_detector = load_ood_detector(model_name, ind_name, layer_idx)

    ind_training_features = get_inter_outputs(model_name, ind_name, ind_name, layer_idx, is_training_set=True)
    ind_testing_features = get_inter_outputs(model_name, ind_name, ind_name, layer_idx)
    ind_length = len(ind_testing_features)

    if ood_name == "combined":
        ood_features = None
        for i in range(len(OOD_LIST)):
            if i == 0:
                ood_features = get_inter_outputs(model_name, ind_name, OOD_LIST[i], layer_idx)
            else:
                ood_features = \
                    np.concatenate((ood_features, get_inter_outputs(model_name, ind_name, OOD_LIST[i], layer_idx)))
        ood_length = len(ood_features)
    else:
        ood_features = get_inter_outputs(model_name, ind_name, ood_name, layer_idx)
        ood_length = len(ood_features)
        ind_length = ood_length = min(ind_length, ood_length)   # balanced testing set

    ind_testing_features = ind_testing_features[:ind_length]
    ood_features = ood_features[:ood_length]

    # Standardize the features
    ss = StandardScaler()
    ss.fit(ind_training_features)
    ind_testing_features = ss.transform(ind_testing_features)
    ood_features = ss.transform(ood_features)

    data = np.vstack((ind_testing_features, ood_features))

    preds = ood_detector.predict(data)

    id_correct = np.count_nonzero(preds[:ind_length] == 1)  # ID is positive for OCSVM
    ood_correct = np.count_nonzero(preds[ind_length:] == -1)    # OOD is negative for OCSVM

    # calculate the average and std of OOD samples' scores
    scores = -ood_detector.decision_function(data)  # negated scores, positive for OOD, negative for InD

    # calculate statistics for current layer
    y_test = [-1] * ind_length + [1] * ood_length
    auroc, fpr_at_95_tpr, aupr_out, aupr_in = get_statistics(y_test, scores)
    print(f'layer {layer_idx}:\t{100*auroc:.2f}')

    # Save results to file
    row = [layer_idx + 1, f'{100*fpr_at_95_tpr:.2f}', f'{100*auroc:.2f}', f'{100*aupr_out:.2f}', f'{100*aupr_in:.2f}',
           id_correct, ood_correct]
    with open(result_filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

    return scores, ind_length, ood_length


if __name__ == '__main__':
    detect_oods_for_all_ood_datasets()
