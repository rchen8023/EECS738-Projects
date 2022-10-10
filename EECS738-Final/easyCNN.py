import torch.nn as nn
import torch
import sys
from matplotlib.cm import get_cmap
jet = get_cmap('jet')


class CNN(nn.Module):

    def __init__(self, class_num):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=3))

        self.feature1 = nn.Sequential()
        self.feature1.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature1.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature1.add_module('f_relu1', nn.ReLU(True))

        self.feature2 = nn.Sequential()
        self.feature2.add_module('f_conv2', nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1)))
        self.feature2.add_module('f_conv3', nn.Conv2d(64, 64, kernel_size=3))
        self.feature2.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature2.add_module('f_drop1', nn.Dropout2d())
        self.feature2.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature2.add_module('f_relu2', nn.ReLU(True))

        self.feature3 = nn.Sequential()
        self.feature3.add_module('f_conv4', nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1)))
        self.feature3.add_module('f_conv5', nn.Conv2d(128, 128, kernel_size=3))
        self.feature3.add_module('f_bn3', nn.BatchNorm2d(128))
        self.feature3.add_module('f_drop3', nn.Dropout2d())
        self.feature3.add_module('f_pool3', nn.MaxPool2d(2))
        self.feature3.add_module('f_relu3', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128 * 2 * 2, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, class_num))

    def forward(self, input_data):
        input_data = input_data.expand(len(input_data), 3, 32, 32)
        data_after_conv1 = self.conv1(input_data)
        feature1 = self.feature1(data_after_conv1)
        feature2 = self.feature2(feature1)
        feature = self.feature3(feature2)
        feature = feature.view(-1, 128 * 2 * 2)
        class_output = self.class_classifier(feature)

        return class_output

    def get_heatmap(self, _input, _label, cmap=jet):
        from trojanvision.utils import apply_cmap

        squeeze_flag = False
        if _input.dim() == 3:
            _input = _input.unsqueeze(0)    # (N, C, H, W)
            squeeze_flag = True
        if isinstance(_label, int):
            _label = [_label] * len(_input)
        _label = torch.as_tensor(_label, device=_input.device)
        heatmap = _input
        _input.requires_grad_()
        _output = self(_input).gather(dim=1, index=_label.unsqueeze(1)).sum()
        grad = torch.autograd.grad(_output, _input)[0]  # (N,C,H,W)
        zero = torch.zeros_like(grad)
        grad = torch.where(grad < 0, zero, grad)
        _input.requires_grad_(False)

        heatmap = grad.abs().max(dim=1)[0]  # (N,H,W)
        heatmap.sub_(heatmap.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
        heatmap.div_(heatmap.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0])

        heatmap = apply_cmap(heatmap.detach().cpu(), cmap)
        return heatmap[0] if squeeze_flag else heatmap
