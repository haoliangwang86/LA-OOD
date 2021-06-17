import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
from global_settings import *
from utility import save_data_hdf5


def main():
    generate_dataset('cifar10', is_training_set=True)
    generate_dataset('cifar10')
    generate_dataset('cifar100', is_training_set=True)
    generate_dataset('cifar100')

    for ood_name in OOD_LIST:
        generate_dataset(ood_name)

    print('All datasets have been successfully generated!')


def generate_dataset(ds_name, is_training_set=False):
    save_dir = 'datasets'
    if is_training_set:
        sample_size = IND_SAMPLE_SIZE
    else:
        sample_size = OOD_SAMPLE_SIZE
    data_loader = get_data_loader(ds_name, is_training_set, batch_size=sample_size, sample_size=sample_size)

    if len(data_loader.dataset) < sample_size:
        sample_size = len(data_loader.dataset)

    print(f'{ds_name} number of available samples: {len(data_loader.dataset)}')
    print(f'{ds_name} number of outputting samples: {sample_size}')

    features, _ = next(iter(data_loader))   # B x C x H x W

    # Convert to numpy array with shape B x H x W x C and value range [0, 255]
    features = features.permute([0, 2, 3, 1])
    features = features.numpy() * 255

    # save data to hdf5 file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_training_set:
        save_data_hdf5(features, "data", f"{save_dir}/{ds_name}_train", "a")
    else:
        save_data_hdf5(features, "data", f"{save_dir}/{ds_name}", "a")

    print(f"{ds_name} dataset successfully generated!")
    print()


def get_data_loader(ds_name, is_training=False,
                    height=IMAGE_HEIGHT, width=IMAGE_WIDTH,
                    batch_size=OOD_SAMPLE_SIZE, sample_size=OOD_SAMPLE_SIZE):
    data_loader = None

    if ds_name == "cifar10":
        data_loader = DataLoader(
            datasets.CIFAR10(root='./data', train=is_training, transform=transforms.Compose([
                transforms.ToTensor(),
            ]), download=True),
            batch_size=batch_size)

    elif ds_name == "cifar100":
        data_loader = DataLoader(
            datasets.CIFAR100(root='./data', train=is_training, transform=transforms.Compose([
                transforms.ToTensor(),
            ]), download=True),
            batch_size=batch_size)

    elif ds_name == "pure_color":
        data_loader = DataLoader(
            PureColorDataset(height=height, width=width, size=sample_size, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=batch_size)

    elif ds_name == "dtd":
        data_loader = DataLoader(
            DTDDataset(transform=transforms.Compose([
                transforms.Resize([32, 32], interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor(),
            ])),
            batch_size=batch_size)

    elif ds_name == "svhn":
        data_loader = DataLoader(
            datasets.SVHN(root='./data', split='test', transform=transforms.Compose([
                transforms.ToTensor(),
            ]), download=True),
            batch_size=batch_size,
            shuffle=True)   # shuffle

    elif ds_name == "tiny":
        data_loader = DataLoader(
            datasets.ImageFolder(root='./data/tiny-imagenet-200/test', transform=transforms.Compose([
                transforms.Resize([32, 32], interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor(),
            ])),
            batch_size=batch_size)

    elif ds_name == "lsun":
        data_loader = DataLoader(
            datasets.LSUN(root='./data', classes='test', transform=transforms.Compose([
                transforms.Resize([32, 32], interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor(),
            ])),
            batch_size=batch_size)

    return data_loader


class PureColorDataset(Dataset):
    """Synthetic pure color dataset."""

    def __init__(self, height, width, size, transform=None):
        self.height = height
        self.width = width
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.ones(self.height, self.width, 3) * torch.rand(3)[None, None, :] * 255
        img = img.numpy().astype(np.uint8)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, 1    # pseudo target


class DTDDataset(Dataset):
    """DTD dataset."""

    def __init__(self, transform=None):
        self.transform = transform
        self.images = []

        directory = 'data/dtd/images'
        for subdir in os.listdir(directory):
            for filename in os.listdir(os.path.join(directory, subdir)):
                if filename.endswith('.jpg'):
                    temp = Image.open(os.path.join(directory, subdir, filename))
                    image = temp.copy()
                    self.images.append(image)
                    temp.close()

        self.size = len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        return img, 1    # pseudo target


if __name__ == '__main__':
    main()
