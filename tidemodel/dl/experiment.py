"""
模型训练, 测试和导出
"""

import copy
import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import accelerate
import torch
import numpy as np
import torch.onnx as onnx
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .callback import Callback, CallbackManager, EarlyStopSaver
from .dataset import collate_fn
from .ops import cross_ic_loss
from ..data import MinuteData, UniqueIndex, cross_ic
from ..ml import Experiment
from ..utils import (
    DummyClass,
    get_logger,
    get_newest_ckpt,
    reset_logger,
)


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


class DLExperiment(Experiment):
    """
    基于Accelerate的深度学习实验基类
    """

    params: Dict[str, Any] = {
        "seed": 42,

        "lr": 0.001,
        "weight_decay": 0.0,

        "batch_size": 128,
        "n_worker": 4,

        "epoch": 100,
        "clip_norm": 5.0,
    
        "n_valid_per_epoch": 1,
        "metric_name": "cross_ic",
        "patience": 5,
    }

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        callbacks: List[Callback] = [],
        params: Dict[str, Any] = {},
    ) -> None:
        _params = self.params.copy()
        _params.update(params)
        self.params = _params

        # 初始化accelerator
        accelerate.utils.set_seed(self.params["seed"])
        self.accelerator = Accelerator(
            dataloader_config=accelerate.utils.DataLoaderConfiguration(
                non_blocking=True
            )
        )

        # 初始化文件夹
        self.folder: str = folder
        if self.accelerator.is_main_process:
            if create_folder:
                # 创建文件夹
                if os.path.exists(folder):
                    shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder)

            self.logger: logging.Logger = get_logger(folder)
            self.logger.info(f"params are {self.params}")

            self.writer = SummaryWriter(folder)
        else:
            self.logger = DummyClass()
            self.writer = DummyClass()
    
        # 注册回调
        self.callback = CallbackManager(self)
        self.callback.add(EarlyStopSaver(
            metric_name=self.params["metric_name"],
            patience=self.params["patience"],
        ))

        for cb in callbacks:
            self.callback.add(cb)

        # 数据
        self.x_names: List[str] = None
        self.y_names: List[str] = None

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

    def _record(self, value: Dict[str, Any], prefix: str) -> None:
        """
        将一个字典的值写入到log和tensorboard中
        """
        self.logger.info(f"step {self.step} {prefix} {dict(value)}")

        for k, v in value.items():
            self.writer.add_scalar(f"{prefix}_{k}", v, self.step)

    def get_model(self, ) -> nn.Module:
        """
        获取原始module
        """
        return self.model.module if hasattr(
            self.model, "module"
        ) else self.model

    def load_data(
        self,
        x_names: List[str],
        y_names: str | List[str],
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        self.x_names: List[str] = x_names
        self.y_names: List[str] = [y_names, ] if isinstance(
            y_names, str
        ) else y_names
        self.train_dataset: Dataset = train_dataset
        self.valid_dataset: Dataset = valid_dataset
        self.test_dataset: Dataset = test_dataset
        self.logger.info(
            f"{len(self.train_dataset)} train, "
            f"{len(self.valid_dataset)} valid, "
            f"{len(self.test_dataset)} test"
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
        if hasattr(self.get_model(), "loss_func"):
            return self.get_model().loss_func(self, data, output)

        return {"loss": cross_ic_loss(data["y"], output["y_pred"], dim=-2)}

    def metric_func(
        self,
        dataset: Dataset,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算指标

        返回一个metric字典

        默认使用整体ic作为评估指标
        """
        if hasattr(self.get_model(), "metric_func"):
            return self.get_model().metric_func(self, dataset, output)
        
        y_pred: MinuteData = copy.copy(dataset.y)
        y_pred.data = output["y_pred"].detach().numpy().reshape(
            dataset.y.shape
        )
        return {"cross_ic": cross_ic(dataset.y, y_pred)}

    def train(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
    ) -> None:
        """
        训练
        """
        torch.cuda.empty_cache()

        self.model = model
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params["lr"],
                weight_decay=self.params["weight_decay"],
            )

        dataloader_config: Dict[str, Any] = {
            "batch_size": self.params["batch_size"],
            "collate_fn": collate_fn,
            "num_workers": self.params["n_worker"],
            "pin_memory": True,
        }
        # 对于rdma挂载不预取更稳定
        if dataloader_config["num_workers"] > 0:
            dataloader_config["prefetch_factor"] = 1
            dataloader_config["persistent_workers"] = True

        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            **dataloader_config,
        )
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            **dataloader_config,
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
        ) // self.params["n_valid_per_epoch"]

        train_loss_agg = LossAggregator()

        self.callback.run("on_train_start")

        # 训练循环
        for e in tqdm(
            range(self.params["epoch"]),
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
                if self.params["clip_norm"] != float("inf"):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.params["clip_norm"],
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
                    if self.accelerator.is_main_process:
                        self.metric = self.metric_func(
                            self.valid_dataloader.dataset, self.output
                        )

                    # 将metric广播到各个节点
                    payload = [self.metric] if self.accelerator.is_main_process else [None]
                    broadcast_object_list(payload, from_process=0)
                    self.metric = payload[0]

                    # 记录
                    self._record(train_loss_agg.loss, prefix="train")
                    train_loss_agg.reset()
                    self._record(valid_loss_agg.loss, prefix="valid")
                    self._record(self.metric, prefix="valid")

                    self.callback.run("on_valid_end")

            self.callback.run("on_epoch_end")

        self.callback.run("on_train_end")

    def test(self, model: nn.Module, ckpt: str | None = None) -> Dict[str, Any]:
        """
        计算测试集指标
        """
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        if ckpt is None:
            ckpt = get_newest_ckpt(self.folder)

        state = torch.load(os.path.join(
            self.folder, ckpt
        ), map_location="cpu")
        self.model = model
        self.model.load_state_dict(state, strict=True)

        dataloader_config: Dict[str, Any] = {
            "batch_size": self.params["batch_size"],
            "collate_fn": collate_fn,
            "num_workers": self.params["n_worker"],
            "pin_memory": True,
        }
        # 对于rdma挂载不预取更稳定
        if dataloader_config["num_workers"] > 0:
            dataloader_config["prefetch_factor"] = 1
            dataloader_config["persistent_workers"] = True

        self.test_dataloader = DataLoader(
            self.test_dataset,
            **dataloader_config,
        )

        self.model, self.test_dataloader = self.accelerator.prepare(
            self.model, self.test_dataloader,
        )

        # 计算输出并保存
        self.output, _ = self.predict(self.test_dataloader)

        if not self.accelerator.is_main_process:
            return {}

        y: MinuteData = self.test_dataloader.dataset.y
        np.save(os.path.join(
            self.folder, "y_pred.npy"
        ), self.output["y_pred"].reshape(*y.data.shape)[:, :, :, 0])
        np.save(os.path.join(
            self.folder, "y.npy"
        ), y.data[:, :, :, 0])

        # 记录指标并打印
        self.metric: Dict[str, float] = self.metric_func(
            self.test_dataloader.dataset, self.output
        )
        self.logger.info(f"test metric: {self.metric}")
        return self.metric

    def predict(
        self,
        dataloader: DataLoader,
    ) -> Tuple[Dict[str, torch.Tensor], LossAggregator]:
        """
        对指定dataloader推理得到输出和聚合loss

        如果数据中不包含y则聚合loss为None

        只聚合y_pred, 防止显存爆炸
        """
        self.model.eval()

        loss_agg = LossAggregator()
        idx_list: List[torch.Tensor] = []
        y_pred_list: List[torch.Tensor] = []

        with torch.inference_mode():
            for data in tqdm(
                dataloader,
                disable=not self.accelerator.is_main_process,
            ):
                output: Dict[str, torch.Tensor] = self.model(data)

                if "y" in data:
                    loss_agg.update(self.loss_func(data, output))

                idx_list.append(data["idx"].reshape(-1).detach().cpu())
                y_pred_list.append(output["y_pred"].detach().cpu())

            # 合并后将输出落盘以避免大矩阵的多机通信
            idx: torch.Tensor = torch.cat(idx_list, axis=0)
            y_pred: torch.Tensor = torch.cat(y_pred_list, axis=0)
            torch.save(
                {"idx": idx, "y_pred": y_pred},
                os.path.join(
                    self.folder,
                    f"tmp_{self.accelerator.process_index}.pth",
                )
            )
            self.accelerator.wait_for_everyone()

            if not self.accelerator.is_main_process:
                return {}, loss_agg

            # 读取并合并y_pred
            idx_list = []
            y_pred_list = []
            
            for i in range(self.accelerator.num_processes):
                tmp: Dict[str, torch.Tensor] = torch.load(
                    os.path.join(self.folder, f"tmp_{i}.pth")
                )
                idx_list.append(tmp["idx"])
                y_pred_list.append(tmp["y_pred"])

            idx = torch.cat(idx_list, axis=0)
            y_pred = torch.cat(y_pred_list, axis=0)

            # 根据idx对output调整顺序和去重并放置到cpu中
            perm: torch.Tensor = torch.argsort(idx)
            idx_sorted: torch.Tensor = idx.index_select(0, perm)
            mask = torch.ones_like(idx_sorted, dtype=torch.bool)
            mask[1: ] = (idx_sorted[1: ] != idx_sorted[: -1])
            y_pred = y_pred.index_select(0, perm[mask]).cpu()
            return {"y_pred": y_pred}, loss_agg

    def export(self, model: nn.Module, ckpt: str | None = None) -> None:
        """
        导出ONNX
        """
        if not self.accelerator.is_main_process:
            return

        torch.cuda.empty_cache()

        if ckpt is None:
            ckpt = get_newest_ckpt(self.folder)

        state = torch.load(os.path.join(
            self.folder, ckpt
        ), map_location="cpu")
        self.model = model
        self.model.to("cpu")
        self.model.load_state_dict(state, strict=True)

        class _ExportOnly(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model: nn.Module = model

            def forward(self, x) -> torch.Tensor:
                return self.model({"x": x})["y_pred"]

        model = _ExportOnly(self.get_model())
        dummy: torch.Tensor = torch.randn(128, 5238, model.model.raw_dim)

        onnx.export(
            model,
            dummy,
            os.path.join(self.folder, "model.onnx"),
            opset_version=17, 
            do_constant_folding=True,
            input_names=["x"],
            output_names=["y_pred"],
            dynamic_axes={"x": {0: "datetime", 1: "ticker", 2: "feat"}}
        )

    def close(self, ) -> None:
        """
        主动调用以释放资源
        """
        if self.accelerator.is_main_process:
            reset_logger(self.logger)
            self.writer.close()


class MinuteDLExperiment(DLExperiment):
    """
    针对一分钟频的深度学习实验基类
    """

    def metric_func(
        self,
        dataset: Dataset,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        返回整体IC, 前30分钟和前90分钟IC
        """
        first_y: MinuteData = copy.copy(dataset.y)
        first_y.data = dataset.y.data[:, :, :, 0: 1]
        first_y.names = UniqueIndex(dataset.y.names.data[0: 1])

        first_y_pred: MinuteData = copy.copy(first_y)
        first_y_pred.data = output["y_pred"].detach().numpy().reshape(
            dataset.y.shape
        )[:, :, :, 0: 1]

        cross_ics: np.ndarray = cross_ic(
            first_y, first_y_pred, mean=False
        )
        min30_idx_slice: slice = first_y.minutes.index_range(None, 30)
        min90_idx_slice: slice = first_y.minutes.index_range(None, 90)
        return {
            "cross_ic": np.mean(cross_ics),
            "cross_ic30": np.mean(cross_ics[:, min30_idx_slice]),
            "cross_ic90": np.mean(cross_ics[:, min90_idx_slice]),
        }
