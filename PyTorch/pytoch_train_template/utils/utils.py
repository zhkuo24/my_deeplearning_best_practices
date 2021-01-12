# -*- coding: utf-8 -*-
# @File   : utils.py
# @Author : zhkuo
# @Time   : 2020/12/18 10:23
# @Desc   :

import os
import random
import numpy as np
import torch
import cv2
# from config import cfg


def cv2_imread(path, mode=cv2.IMREAD_COLOR):
    """
    read image based on cv2 mode
    """
    img_read = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    return img_read


def cv2_imwrite(path, img_write):
    """
    save image based on cv2 mode
    """
    suffix = os.path.splitext(path)[-1]
    cv2.imencode(suffix, img_write)[1].tofile(path)


def get_img(path):
    im_bgr = cv2_imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # can change the flag
    # ref: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


