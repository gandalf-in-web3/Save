"""
机器学习实验基类
"""

import logging
import os
import shutil
from typing import Any, Dict

from abc import ABC, abstractmethod

from ..utils import get_logger, reset_logger, set_seed


class Experiment(ABC):
    """
    实验基类
    
    每个实验对应一个文件夹

    训练和测试的相关参数均在初始化时通过params传入
    """

    params: Dict[str, Any] = {
        "seed": 42,
    }

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        params: Dict[str, Any] = {},
    ) -> None:
        self.folder: str = folder

        _params = self.params.copy()
        _params.update(params)
        self.params = _params

        set_seed(self.params["seed"])

        if create_folder:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)

        self.logger: logging.Logger = get_logger(folder)
        self.logger.info(f"params are {self.params}")

    @abstractmethod
    def load_data(self, *args, **kwargs) -> None:
        """
        加载数据集
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """
        训练
        """
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> None:
        """
        测试
        """
        pass
    
    def close(self, ) -> None:
        """
        释放资源
        """
        reset_logger(self.logger)
