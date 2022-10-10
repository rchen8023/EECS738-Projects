import torch.nn as nn
import torch
import sys


class CNN(nn.Module):

    def __init__(self, source_num, target_num):
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
        self.class_classifier.add_module('c_fc2', nn.Linear(100, source_num))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(source_num, target_num))


    def forward(self, input_data):
        input_data = input_data.expand(len(input_data), 3, 32, 32)
        data_after_conv1 = self.conv1(input_data)
        feature1 = self.feature1(data_after_conv1)
        feature2 = self.feature2(feature1)
        feature = self.feature3(feature2)
        feature = feature.view(-1, 128 * 2 * 2)
        class_output = self.class_classifier(feature)

        return class_output
