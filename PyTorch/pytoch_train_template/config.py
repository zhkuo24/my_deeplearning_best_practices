# -*- coding: utf-8 -*-
# @File   : config.py
# @Author : zhkuo
# @Time   : 2020/12/17 17:28
# @Desc   :

import os
import sys
import torch
from loguru import logger
from tomlkit import parse
import time


# default_std_out_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
default_std_out_format = "{level.icon} <level>{message}</level>"
default_file_format = "{time:YYYY-MM-DD HH:mm:ss} {level} {message}"


class Config:
    def __init__(self):
        """
        set the config
        """
        self.cfg = None
        self.zlogger = None

    def read_cfg(self, cfg_path):
        """
        read the config from toml file
        """
        with open(cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.cfg = parse(content)

        return self.cfg

    def set_logger(self, std_out_format=default_std_out_format, file_format=default_file_format):
        """
        init the logger
        """
        if self.cfg['common']['log_file_use_time']:
            nowtime_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        else:
            nowtime_str = ""
        if self.cfg['common']['model_name'] == "":
            model_name = "model"
        else:
            model_name = self.cfg['common']['model_name']
        log_fname = f"{model_name}_ver{self.cfg['common']['version']}_{nowtime_str}.log"
        log_file = os.path.join(self.cfg['path']['log_path'], log_fname)

        logger.remove()
        logger.level("DL", no=12, color="<blue>", icon="👽")
        logger.add(sys.stdout, colorize=True, format=std_out_format)
        if self.cfg['common']['log_file']:
            if os.path.exists(log_file):
                os.remove(log_file)
            logger.add(log_file, format=file_format)

        def custom_logger(msg, lev="DL"):
            return logger.log(lev, msg)
        self.zlogger = custom_logger
        return self.zlogger

    def show_ver_info(self):
        self.zlogger(f"torch version: {torch.__version__}")
        if torch.cuda.is_available():
            self.zlogger(f"cuda version: {torch.version.cuda}")
            self.zlogger(f"cudnn version: {torch.backends.cudnn.version()}")
            self.zlogger(f"GPU: {torch.cuda.get_device_name(0)}")


# config and version info
cfg_cls = Config()
cfg = cfg_cls.read_cfg('config.toml')
zlogger = cfg_cls.set_logger()
cfg_cls.show_ver_info()
