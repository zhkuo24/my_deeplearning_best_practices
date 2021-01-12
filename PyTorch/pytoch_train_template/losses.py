# -*- coding: utf-8 -*-
# @File   : losses.py
# @Author : zhkuo
# @Time   : 2020/12/18 10:34
# @Desc   : define the loss of net, can be custom


import torch.nn as nn


loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.CrossEntropyLoss().to(device)
