"""
模型训练, 测试和导出
"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import accelerate
import torch
import numpy as np
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .callback import Callback, CallbackManager
from .dataset import HDF5MinuteDataset, collate_fn
from .ops import cross_ic_loss
from ..data import HDF5MinuteDataBase, cross_ic
from ..utils import DummyClass, get_logger, get_newest_ckpt, reset_logger


class LossAggregator:
    """
    以均值的方式聚合loss
    """

    def __init__(self, ) -> None:
        self.num: int = 0
        self.loss: Dict[str, float] = defaultdict(int)

    def reset(self, ) -> None:
        """
        重置
        """
        self.num = 0
        self.loss = defaultdict(int)

    def update(self, loss: Dict[str, torch.Tensor]) -> None:
        """
        更新
        """
        self.num += 1

        for k, v in loss.items():
            self.loss[k] += (v.item() - self.loss[k]) / self.num


class DLExperiment(ABC):
    """
    基于Accelerate编写的模型实验基类
    """

    config: Dict[str, Any] = {
        "seed": 42,
        "folder": None,
        "create_folder": True,

        "lr": 0.001,
        "weight_decay": 0.001,

        "batch_size": 128,
        "n_worker": 4,

        "epoch": 100,
        "clip_norm": float("inf"),
        "n_valid_per_epoch": 1,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        callbacks: List[Callback] = [],
    ) -> None:
        _config = self.config.copy()
        _config.update(config)
        self.config = _config
        
        # 注册回调
        self.callback = CallbackManager(self)
        for cb in callbacks:
            self.callback.add(cb)

        # 初始化accelerator
        accelerate.utils.set_seed(self.config["seed"])
        self.accelerator = Accelerator(
            dataloader_config=accelerate.utils.DataLoaderConfiguration(
                non_blocking=True
            )
        )

        # 初始化文件夹
        self._init_folder()

        # 数据
        self.train_dataset: Dataset | None = None
        self.valid_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self.train_dataloader: DataLoader | None = None
        self.valid_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None

        # 模型
        self.model: nn.Module = None
        self.optimizer: Optimizer = None

        # 运行变量
        self.epoch: int = 0
        self.step: int = 0
        self.valid_step: int = 0
        self.stop_train: bool = False

        self.output: Dict[str, torch.Tensor] = {}
        self.metric: Dict[str, torch.Tensor] = {}

    def _init_folder(self, ) -> None:
        """
        创建文件夹, 日志和tensorboard
        """
        if self.accelerator.is_main_process:
            if self.config["create_folder"]:
                # 创建文件夹
                if os.path.exists(self.config["folder"]):
                    shutil.rmtree(self.config["folder"], ignore_errors=True)
                os.makedirs(self.config["folder"])

            self.logger: logging.Logger = get_logger(self.config["folder"])
            self.logger.info(f"config is {self.config}")

            self.writer = SummaryWriter(self.config["folder"])
        else:
            self.logger = DummyClass()
            self.writer = DummyClass()

    def _record(self, value: Dict[str, Any], prefix: str) -> None:
        """
        将一个字典的值写入到log和tensorboard中
        """
        self.logger.info(f"step {self.step} {prefix} {dict(value)}")

        for k, v in value.items():
            self.writer.add_scalar(f"{prefix}_{k}", v, self.step)

    @abstractmethod
    def load_data(self, ) -> None:
        """
        加载Dataset
        """
        pass

    @abstractmethod
    def build_model(self, ) -> nn.Module:
        """
        构造并返回模型

        模型的forward函数返回一个字典, 其中y_pred键值为最终预测值
        """
        pass
    
    def build_optimizer(self, ) -> Optimizer:
        """
        构造并返回优化器

        默认使用Adam优化器
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
    
    def loss_func(
        self,
        data: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失函数

        返回一个字典, 其中loss键值用于反向传播的loss

        默认使用单label cross ic loss
        """
        return {"loss": cross_ic_loss(
            data["y"].squeeze(-1),
            output["y_pred"].squeeze(-1)
        )}

    def metric_func(
        self,
        dataset: Dataset,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算指标

        返回一个metric字典

        默认使用ic作为评估指标, 包含整体IC, 前30分钟和前90分钟IC
        """
        dataset.y_pred.data = output["y_pred"].detach().numpy().reshape(
            dataset.y_pred.shape
        )
        cross_ics: np.ndarray = cross_ic(dataset.y, dataset.y_pred, mean=False)
        min30_idx_slice: slice = dataset.y.minutes.index_range(None, 30)
        min90_idx_slice: slice = dataset.y.minutes.index_range(None, 90)
        return {
            "cross_ic": np.mean(cross_ics),
            "cross_ic30": np.mean(cross_ics[:, min30_idx_slice]),
            "cross_ic90": np.mean(cross_ics[:, min90_idx_slice]),
        }

    def train(self, ) -> None:
        """
        训练
        """
        torch.cuda.empty_cache()

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

        # 对于rdma挂载不预取更稳定
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config["n_worker"],
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config["n_worker"],
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False,
        )

        # accelerator处理变量
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.valid_dataloader
        ) = (
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.valid_dataloader,
            )
        )

        self.valid_step = len(
            self.train_dataloader
        ) // self.config["n_valid_per_epoch"]

        train_loss_agg = LossAggregator()

        self.callback.run("on_train_start")

        # 训练循环
        for e in tqdm(
            range(self.config["epoch"]),
            disable=not self.accelerator.is_main_process,
        ):
            # 终止训练
            if self.stop_train:
                break

            self.epoch = e

            self.callback.run("on_epoch_start")

            for data in tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_main_process,
            ):
                # 终止训练
                if self.stop_train:
                    break

                self.model.train()
                self.callback.run("on_train_step_start")

                # 正向
                output: Dict[str, torch.Tensor] = self.model(data)
                loss: Dict[str, torch.Tensor] = self.loss_func(data, output)
                train_loss_agg.update(loss)
                
                self.accelerator.backward(loss["loss"])
                if self.config["clip_norm"] != float("inf"):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config["clip_norm"]
                    )

                # 反向
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.step += 1
                self.callback.run("on_train_step_end")

                # 验证
                if self.step % self.valid_step == 0:
                    self.callback.run("on_valid_start")

                    # 正向
                    self.output, valid_loss_agg = self.predict(
                        self.valid_dataloader
                    )
                    self.metric = self.metric_func(
                        self.valid_dataloader.dataset, self.output
                    )

                    # 记录
                    self._record(train_loss_agg.loss, prefix="train")
                    train_loss_agg.reset()

                    self._record(valid_loss_agg.loss, prefix="valid")
                    self._record(self.metric, prefix="valid")

                    self.callback.run("on_valid_end")

            self.callback.run("on_epoch_end")

        self.callback.run("on_train_end")

    def test(self, ckpt: str | None = None) -> None:
        """
        计算测试集指标
        """
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        if ckpt is None:
            ckpt = get_newest_ckpt(self.config["folder"])

        state = torch.load(os.path.join(
            self.config["folder"], ckpt
        ), map_location="cpu")
        self.model = self.build_model()
        self.model.load_state_dict(state, strict=True)

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config["n_worker"],
            prefetch_factor=1,
            persistent_workers=False,
            pin_memory=True,
            drop_last=False,
        )

        self.model, self.test_dataloader = self.accelerator.prepare(
            self.model, self.test_dataloader,
        )

        # 计算输出并保存
        self.output, _ = self.predict(self.test_dataloader)

        if self.accelerator.is_main_process:
            np.save(os.path.join(
                self.config["folder"], "y_pred.npy"
            ), self.output["y_pred"])
            np.save(os.path.join(
                self.config["folder"], "y.npy"
            ), self.test_dataloader.dataset.y.data)

        # 记录指标并打印
        self.metric: Dict[str, float] = self.metric_func(
            self.test_dataloader.dataset, self.output
        )
        self.logger.info(f"test metric: {self.metric}")

    def predict(
        self,
        dataloader: DataLoader,
    ) -> Tuple[torch.Tensor, LossAggregator]:
        """
        对指定dataloader推理得到输出和聚合loss

        如果数据中不包含y则聚合loss为None
        """
        self.model.eval()

        loss_agg = LossAggregator()
        idx_list: List[torch.Tensor] = []
        output_list: List[Dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for data in tqdm(
                dataloader,
                disable=not self.accelerator.is_main_process,
            ):
                output: Dict[str, torch.Tensor] = self.model(data)

                if "y" in data:
                    loss_agg.update(self.loss_func(data, output))

                idx_list.append(data["idx"].reshape(-1))
                output_list.append(output)

            idx: torch.Tensor = torch.cat(idx_list, axis=0)
            idx = self.accelerator.gather_for_metrics(idx)

            output: Dict[str, torch.Tensor] = {}
            for k in ["y_pred", ]:
                output[k] = torch.cat(
                    [output_list[i][k] for i in range(len(output_list))
                ], axis=0)
                output[k] = self.accelerator.gather_for_metrics(output[k])

            # 根据idx对output调整顺序和去重并放置到cpu中
            perm: torch.Tensor = torch.argsort(idx)
            idx_sorted: torch.Tensor = idx.index_select(0, perm)

            mask = torch.ones_like(idx_sorted, dtype=torch.bool)
            mask[1:] = (idx_sorted[1:] != idx_sorted[:-1])

            for k in output:
                output[k] = output[k].index_select(0, perm[mask]).cpu()
            return output, loss_agg

    def close(self, ) -> None:
        """
        主动调用以释放资源
        """
        if self.accelerator.is_main_process:
            reset_logger(self.logger)
            self.writer.close()


class HDF5DLExperiment(DLExperiment):
    """
    基于HDF5数据库的DL实验
    """

    config: Dict[str, Any] = {
        **DLExperiment.config,

        "bin_folder": None,
        "hdf5_folder": None,
        
        "x_names": slice(None),
        "y_names": None,
        "start_minute": None,
        "end_minute": None,
        "stride": 1,
        "seq_len": None,

        "train_start_dt": None,
        "train_end_dt": None,
        "valid_start_dt": None,
        "valid_end_dt": None,
        "test_start_dt": None,
        "test_end_dt": None,
    }

    def load_data(self, ) -> None:
        """
        从HDF5数据库中加载数据
        """

        hdf5_db = HDF5MinuteDataBase(
            self.config["hdf5_folder"], self.config["bin_folder"]
        )
        if self.config["x_names"] == slice(None):
            self.x_names: List[str] = hdf5_db.names
        else:
            self.x_names = self.config["x_names"]

        dataset_cls: Callable = partial(
            HDF5MinuteDataset,
            hdf5_db=hdf5_db,
            x_names=self.config["x_names"],
            y_names=self.config["y_names"],
            start_minute=self.config["start_minute"],
            end_minute=self.config["end_minute"],
            seq_len=self.config["seq_len"],
        )

        self.train_dataset = dataset_cls(
            start_dt=self.config["train_start_dt"],
            end_dt=self.config["train_end_dt"],
            stride=self.config["stride"],
        )
        self.valid_dataset = dataset_cls(
            start_dt=self.config["valid_start_dt"],
            end_dt=self.config["valid_end_dt"],
            stride=self.config["stride"],
        )

        # 测试时步长必须为1
        self.test_dataset = dataset_cls(
            start_dt=self.config["test_start_dt"],
            end_dt=self.config["test_end_dt"],
            stride=1,
        )
        # self.logger.info(
        #     f"{len(self.train_dataset)} train, "
        #     f"{len(self.valid_dataset)} valid, "
        #     f"{len(self.test_dataset)} test"
        # )
