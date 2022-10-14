# LA-OOD
This is a pytorch implementation for LA-OOD: Layer Adaptive Deep Neural Networks for Out-of-distribution Detection.

# Overview
<img width="2186" alt="Overview" src="https://user-images.githubusercontent.com/71032219/122402188-de578180-cf42-11eb-9f0f-9eb45af5292d.png">
An overview of our proposed Layer Adaptive Deep Neural Networks for OOD Detection (LA-OOD). As the deeper layers of a DNN tend to have higher capacity for representing more sophisticated concepts, our framework fully utilizes the intermediate outputs of a DNN to identify OODs of different complexity. Average pooling is done to first reduce the feature dimension of the intermediate outputs, then the low-dimension feature vectors are fed into the OOD detectors to generate OOD scores for the input samples. The OOD detectors are trained in an unsupervised self-adaptive setting, hence LA-OOD can be flexibly applied to any existing DNNs with minor computation and data cost.

# Requirements
* Pyhton 3.7.5
* Pytorch 1.9.0
* CUDA 10.1

Please use the following codes to install the full requirements:
```python
pip install -r requirements.txt
```

# Pre-trained models
|   Model  | CIFAR10 | CIFAR100 |
|:--------:|:-------:|:--------:|
|    VGG   |  93.94%  |   74.13%  |
| DenseNet |  95.06%  |   77.18%  |
|  ResNet  |  94.67%  |   75.02%  |

# Running the codes
## 1. Train the backbone models
If you wish to train a new backbone model, run the following:
```python
python train_backbone_model.py --model vgg16 --dataset cifar10
```

for the backbone models, choose from:
* vgg16
* resnet34
* densenet100

for the training dataset, choose from:
* cifar10
* cifar100

## 2. Download the OOD datasets
Download the following datasets:
* LSUN test set: https://github.com/fyu/lsun
* Tiny ImageNet: https://image-net.org/index.php
* DTD dataset: https://www.robots.ox.ac.uk/~vgg/data/dtd/

save the unzipped files in ./data folder


## 3. Generate all the datasets
Generate the InD and OOD datasets:
```python
python generate_datasets.py
```

## 4. Save the intermedia outputs
```python
python save_inter_outputs.py --model vgg16 --ind cifar10
```

## 5. Train OOD detectors
```python
python train_ood_detectors.py --model vgg16 --ind cifar10
```

## 6. Test: OOD detection
```python
python detect_oods.py --model vgg16 --ind cifar10
```

## 7. Co-train models (optional)
```python
python co_train.py --model vgg16 --dataset cifar10
```
