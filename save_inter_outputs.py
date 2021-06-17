import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from global_settings import *
from utility import save_data_hdf5, CustomDataset
import backbone_models.densenet as densenet
import backbone_models.resnet as resnet
import backbone_models.vgg as vgg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vgg16', type=str,
                    help='model name')
parser.add_argument('--ind', default='cifar10', type=str,
                    help='InD dataset name')


def main():
    args = parser.parse_args()
    save_inter_outputs(args.model, args.ind, args.ind, is_training_set=True)
    save_inter_outputs(args.model, args.ind, args.ind)
    for ood_name in OOD_LIST:
        save_inter_outputs(args.model, args.ind, ood_name)


def save_inter_outputs(model_name, id_name, ds_name=None, is_training_set=False):
    if not is_training_set and ds_name == None:
        print("Must set the test dataset name!")
        return

    save_dir = f"inter_outputs/{model_name}/{id_name}_vs_others"

    if is_training_set:
        file_address = f"{save_dir}/{ds_name}_train"
    else:
        file_address = f"{save_dir}/{ds_name}_test"

    if os.path.exists(f"{file_address}.hdf5"):
        print("Features already exists.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if is_training_set:
        print(f"saving {ds_name} training set outputs ...")
    else:
        print(f"saving {ds_name} testing set outputs ...")

    if is_training_set:
        sample_size = IND_SAMPLE_SIZE
    else:
        sample_size = OOD_SAMPLE_SIZE
    batch_size = BATCH_SIZE

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    data_loader = DataLoader(
            CustomDataset(ds_name=ds_name, is_training=is_training_set, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), sample_size=sample_size),
            batch_size=batch_size)

    if len(data_loader.dataset) < sample_size:
        sample_size = len(data_loader.dataset)

    print(f'{ds_name} number of available samples: {len(data_loader.dataset)}')
    print(f'{ds_name} number of outputting samples: {sample_size}')

    # Load pre-trained model
    if model_name == "vgg16":
        if id_name == "cifar100":
            model = vgg.vgg16_cifar100().cuda()
        elif id_name == "cifar10":
            model = vgg.vgg16().cuda()
    elif model_name == "resnet34":
        if id_name == "cifar100":
            model = resnet.ResNet34_cifar100().cuda()
        else:
            model = resnet.ResNet34().cuda()
    elif model_name == "densenet100":
        if id_name == "cifar100":
            model = densenet.DenseNet100_cifar100().cuda()
        elif id_name == "cifar10":
            model = densenet.DenseNet100().cuda()

    if os.path.exists(f'pre_trained_backbones/{model_name}-{id_name}.h5'):
        print(f'Loading pre-trained {model_name}-{id_name} model ...')
        model.load_state_dict(torch.load(f'pre_trained_backbones/{model_name}-{id_name}.h5'))
        print(f'Pre-trained {model_name}-{id_name} model successfully loaded.')
    else:
        print("Pre-trained model not found!")
        return

    model.cuda().eval()

    with torch.no_grad():
        total = 0
        features = None
        for data, _ in data_loader:
            total += batch_size
            if total > sample_size:
                remains = batch_size - (total - sample_size)
                data = data[:remains]
            data = data.cuda()
            outputs = model.get_inter_outputs(data)  # layers x batch_size x C x H X W

            # get channel mean
            for i in range(len(outputs)):
                if len(outputs[i].shape) == 4:
                    outputs[i] = np.mean(outputs[i], axis=(2, 3))  # batch_size x C

            if features is None:
                features = outputs  # layers x batch_size x C
            else:
                for i in range(len(features)):
                    features[i] = np.vstack((features[i], outputs[i]))  # stack each batch

            if total >= sample_size:
                break

        # save data
        for i in range(len(features)):
            save_data_hdf5(features[i], str(i), file_address, "a")

    if is_training_set:
        print(f"{ds_name} training set intermediate outputs successfully saved!")
    else:
        print(f"{ds_name} testing set intermediate outputs successfully saved!")
    print()


if __name__ == '__main__':
    main()
