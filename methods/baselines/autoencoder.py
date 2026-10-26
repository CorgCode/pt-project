"""Небольшой нейросетевой denoising autoencoder для поведенческих профилей хостов.

Модель учится восстанавливать профиль хоста из слегка зашумлённой копии. Скрытый слой
размерности `dim` используется как эмбеддинг. Обучение не требует меток и читает только train.
Для поздних окон применяется тот же энкодер, поэтому пространство координат общее.

PyTorch — необязательная зависимость проекта; для этого метода установите `pip install torch`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from methods.baselines.temporal import Baseline, BehavioralBaseline, _frame
from pipeline.data import ProcessedData


def _make_network(input_dim: int, hidden_dim: int, embedding_dim: int):
    """Создать сеть внутри функции, чтобы импорт модуля работал и без установленного PyTorch."""
    try:
        import torch  # noqa: F401
        from torch import nn
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "BehavioralDAE требует PyTorch. Установите его командой: pip install torch"
        ) from exc

    class DenoisingAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return DenoisingAutoencoder()


class BehavioralDAE(Baseline):
    """Denoising autoencoder (DAE) по поведенческим признакам хоста.

    В отличие от GraphSAGE, сеть не агрегирует графовых соседей. В отличие от
    temporal_encoder, она не обучается сопоставлять одного хоста в двух временных
    фрагментах: задача обучения — реконструировать поведенческий вектор.
    """

    name = "behavioral_dae"
    shared_coordinate_system = True
    description = (
        "MLP denoising autoencoder trained to reconstruct host behavioural profiles; "
        "latent bottleneck is the embedding; train-only fit, shared encoder for later windows"
    )

    def __init__(
        self,
        dim: int = 16,
        hidden_dim: int = 64,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        noise_std: float = 0.10,
        weight_decay: float = 1e-5,
        seed: int = 17,
    ):
        if dim < 1 or hidden_dim < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("dim, hidden_dim, epochs и batch_size должны быть положительными")
        if lr <= 0 or noise_std < 0 or weight_decay < 0:
            raise ValueError("lr должен быть > 0, noise_std и weight_decay — >= 0")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.noise_std = float(noise_std)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self._behavior: BehavioralBaseline | None = None
        self._net: Any = None
        self.loss_history_: list[float] = []
        self.n_features_in_: int | None = None

    def fit(self, data: ProcessedData) -> "BehavioralDAE":
        import torch
        from torch import nn

        # BehavioralBaseline.fit читает только data.training_flows(). Его fitted-преобразования
        # (порты, протоколы и нормировки признаков) затем фиксируются для всех временных окон.
        self._behavior = BehavioralBaseline().fit(data)
        x_np = self._behavior._features(data, "train", list(data.cohort)).to_numpy(dtype=np.float32)
        if x_np.ndim != 2 or x_np.shape[0] < 2 or x_np.shape[1] < 1:
            raise ValueError(f"behavioral_dae: некорректная матрица train-признаков {x_np.shape}")
        if not np.isfinite(x_np).all():
            raise ValueError("behavioral_dae: train-признаки содержат NaN или бесконечность")

        torch.manual_seed(self.seed)
        # Ограничиваем потоки BLAS/PyTorch ради предсказуемой скорости на CPU.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 2))
        try:
            self._net = _make_network(x_np.shape[1], self.hidden_dim, self.dim)
            optimizer = torch.optim.AdamW(
                self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            loss_fn = nn.MSELoss()
            x = torch.tensor(x_np.tolist(), dtype=torch.float32, device="cpu")
            generator = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            self.loss_history_ = []

            self._net.train()
            for _epoch in range(self.epochs):
                order = torch.randperm(len(x), generator=generator)
                total_loss = 0.0
                seen = 0
                for start in range(0, len(order), self.batch_size):
                    ids = order[start:start + self.batch_size]
                    target = x[ids]
                    noisy = target + self.noise_std * torch.randn(
                        target.shape, generator=generator, dtype=target.dtype
                    )
                    prediction = self._net(noisy)
                    loss = loss_fn(prediction, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    count = len(ids)
                    total_loss += float(loss.detach()) * count
                    seen += count
                # История — MSE на чистых профилях, чтобы отслеживать именно качество
                # восстановления, а не шумовую добавку к входу.
                self._net.eval()
                with torch.no_grad():
                    clean_mse = float(loss_fn(self._net(x), x))
                self.loss_history_.append(clean_mse)
                self._net.train()
        finally:
            torch.set_num_threads(previous_threads)

        self.n_features_in_ = int(x_np.shape[1])
        self.n_hosts_fit_ = int(x_np.shape[0])
        return self

    def embed(
        self,
        data: ProcessedData,
        window: str,
        hosts: list[str] | None = None,
    ) -> pd.DataFrame:
        import torch

        if self._behavior is None or self._net is None:
            raise RuntimeError("сначала вызовите BehavioralDAE.fit(data)")
        hosts = list(hosts) if hosts is not None else list(data.cohort)
        x_np = self._behavior._features(data, window, hosts).to_numpy(dtype=np.float32)
        if x_np.shape[1] != self.n_features_in_:
            raise ValueError(
                f"behavioral_dae: ожидалось {self.n_features_in_} признаков, получено {x_np.shape[1]}"
            )
        if not np.isfinite(x_np).all():
            raise ValueError(f"behavioral_dae: признаки окна {window!r} содержат NaN или бесконечность")
        self._net.eval()
        with torch.no_grad():
            z_tensor = self._net.encoder(torch.tensor(x_np.tolist(), dtype=torch.float32, device="cpu"))
            # Используем .tolist(), а не Tensor.numpy(): это обход несовместимого NumPy bridge
            # в некоторых сборках PyTorch/NumPy на Windows.
            z = np.asarray(z_tensor.cpu().tolist(), dtype=float)
        if not np.isfinite(z).all():
            raise ValueError(f"behavioral_dae: сеть вернула нечисловой эмбеддинг для окна {window!r}")
        return _frame(z, hosts)

    def info(self) -> dict:
        out = super().info()
        out.update({
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "noise_std": self.noise_std,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "n_features_in": self.n_features_in_,
            "final_train_mse": self.loss_history_[-1] if self.loss_history_ else None,
        })
        return out
