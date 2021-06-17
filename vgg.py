'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, fc_size=512):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def intermediate_forward(self, x):
        outputs = []
        for l in list(self.features.modules())[1:]:
            x = l(x)
            if type(l) == nn.Conv2d:
                outputs.append(x.clone())
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        outputs.append(x.clone())
        return outputs

    def get_inter_outputs(self, x):
        outputs = []
        for l in list(self.features.modules())[1:]:
            x = l(x)
            if type(l) == nn.Conv2d:
                outputs.append(x.cpu().numpy())
            # outputs.append(x.cpu().numpy())
        x = x.view(x.size(0), -1)
        # for l in list(self.classifier.modules())[1:]:
        #     x = l(x)
            # if type(l) == nn.Conv2d or type(l) == nn.MaxPool2d:
            #     outputs.append(x.cpu().numpy())
            # outputs.append(x.cpu().numpy())
        x = self.classifier(x)
        outputs.append(x.cpu().numpy())
        return outputs

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def vgg16():
    return VGG('VGG16', 10)

def vgg16_cifar100():
    return VGG('VGG16', 100)

def test():
    net = VGG('VGG16', 10)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()