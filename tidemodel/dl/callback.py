"""
实验回调类
"""

from __future__ import annotations

import os
import time
from abc import ABC
from typing import List, Literal, TYPE_CHECKING

import torch
from ray.air import session
from transformers import get_linear_schedule_with_warmup

if TYPE_CHECKING:
    from .experiment import DLExperiment


class Callback(ABC):
    """
    回调类

    在实验过程中按顺序自动调用执行
    """

    def on_epoch_start(self, exp: DLExperiment) -> None:
        pass

    def on_epoch_end(self, exp: DLExperiment) -> None:
        pass

    def on_train_start(self, exp: DLExperiment) -> None:
        pass

    def on_train_end(self, exp: DLExperiment) -> None:
        pass

    def on_train_step_start(self, exp: DLExperiment) -> None:
        pass

    def on_train_step_end(self, exp: DLExperiment) -> None:
        pass

    def on_valid_start(self, exp: DLExperiment) -> None:
        pass
    
    def on_valid_end(self, exp: DLExperiment) -> None:
        pass


class CallbackManager:
    """
    管理多个回调并执行相应的方法
    """

    def __init__(self, exp: DLExperiment) -> None:
        self.exp: DLExperiment = exp

        self.callbacks: List[Callback] = []

    def add(self, callback: Callback) -> None:
        """
        添加回调
        """
        self.callbacks.append(callback)

    def run(self, name: str) -> None:
        """
        根据传入的函数名执行回调
        """
        for callback in self.callbacks:
            getattr(callback, name)(self.exp)


class EarlyStopSaver(Callback):
    """
    根据metric早停和保存模型
    """

    def __init__(
        self,
        metric_name: str,
        patience: int,
        mode: Literal["min", "max"] = "max",
        tol: float = 0.0,
    ) -> None:
        self.metric_name: str = metric_name
        self.patience: int = patience
        self.mode: Literal["min", "max"] = mode
        self.tol: float = tol

        self.current_patience: int = self.patience
        self.best_metric: float = None

    def _is_better(self, metric: float) -> bool:
        """
        返回新指标是否更优
        """
        if self.best_metric is None:
            return True
        
        if self.mode == "min":
            return metric < self.best_metric - self.tol
        elif self.mode == "max":
            return metric > self.best_metric + self.tol
        else:
            raise ValueError(f"{self.mode} must be in [min, max]")

    def on_valid_end(self, exp: DLExperiment) -> None:
        """
        在一次验证根据指标早停和保存模型
        """
        if self._is_better(exp.metric[self.metric_name]):
            self.best_metric = exp.metric[self.metric_name]

            exp.logger.info(f"step {exp.step} new best model")

            # 保存最优模型
            if exp.accelerator.is_main_process:
                torch.save(
                    exp.get_model().state_dict(),
                    os.path.join(exp.folder, f"model_{exp.step}.pth"),
                )

            self.current_patience = self.patience
        else:
            # 触发早停逻辑
            self.current_patience -= 1
            if self.current_patience == 0:
                exp.logger.info("early stop")
                exp.stop_train = True


class SimpleProfile(Callback):
    """
    通过记录训练耗时分析卡点
    """

    def __init__(self, ) -> None:
        self.start_time: float = None
        self.end_time: float = None

    def on_train_step_start(self, exp: DLExperiment) -> None:
        self.start_time = time.time()

    def on_train_step_end(self, exp: DLExperiment) -> None:
        end_time = time.time()
        
        # 跳过预热时间
        if exp.step > 10:
            exp.logger.info(
                f"data cost {(self.start_time - self.end_time):.3f}s, "
                f"gpu cost {(end_time - self.start_time):.3f}s"
            )

        self.end_time = end_time


class WarmUpSchedule(Callback):
    """
    预热学习率调度器
    """

    def __init__(self, warmup_ratio: float = 0.0) -> None:
        self.warmup_ratio: float = warmup_ratio

    def on_train_start(self, exp: DLExperiment) -> None:
        total_step: int = len(exp.train_dataloader) * exp.params["epoch"]
        exp.scheduler = get_linear_schedule_with_warmup(
            exp.optimizer,
            num_warmup_steps=int(total_step * self.warmup_ratio),
            num_training_steps=total_step,
        )

    def on_train_step_end(self, exp: DLExperiment) -> None:
        exp.scheduler.step()


class RayTuneReport(Callback):
    """
    在每次验证结束时通过ray上报指标

    上报的是到目前为止的最优指标
    """

    def __init__(
        self,
        metric_name: str,
        mode: Literal["min", "max"] = "max",
    ) -> None:
        self.metric_name: str = metric_name
        self.mode: Literal["min", "max"] = mode

        self.best_metric: float = None

    def _is_better(self, metric: float) -> bool:
        """
        返回新指标是否更优
        """
        if self.best_metric is None:
            return True
        
        if self.mode == "min":
            return metric < self.best_metric
        elif self.mode == "max":
            return metric > self.best_metric
        else:
            raise ValueError(f"{self.mode} must be in [min, max]")

    def on_valid_end(self, exp: DLExperiment) -> None:
        if self._is_better(exp.metric[self.metric_name]):
            self.best_metric = exp.metric[self.metric_name]
        
        session.report({self.metric_name: self.best_metric})
