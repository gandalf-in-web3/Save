"""
辅助设置系统环境
"""

import os
import logging
import random

import torch
import numpy as np


def set_seed(seed: int) -> None:
    """
    设置随机数种子
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(
    path: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    设置logger用于在终端和文件中同步输出日志
    """

    logger: logging.Logger = logging.getLogger("main")
    logger.setLevel(level)
    
    # 避免重复设置
    if logger.handlers:
        return logger

    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # 设置文件输出
    file_handler = logging.FileHandler(
        os.path.join(path, "main.log"), encoding="utf-8"
    )
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)

    # 设置终端输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(format)
    logger.addHandler(console_handler)
    return logger
