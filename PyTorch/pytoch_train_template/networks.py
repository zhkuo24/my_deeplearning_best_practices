# -*- coding: utf-8 -*-
# @File   : netmodel.py
# @Author : zhkuo
# @Time   : 2020/12/18 10:33
# @Desc   : define the model of net, custom or hub


import torch.nn as nn
import timm

import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  #
        self.pool = nn.MaxPool2d(2, 2)  #
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 256)  #
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # NLLLoss()
        # x = F.log_softmax(x,dim=1)
        return x


def get_model(model_arch, n_class, pretrained=True, set_freeze=True):
    """
    get the model
    """
    model = timm.create_model(model_arch, pretrained=pretrained)
    if set_freeze:
        for param in model.parameters():
            param.requires_grad = False

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_class)
    return model


if __name__ == '__main__':
    from pprint import pprint

    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
    print(len(model_names))
