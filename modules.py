"""
modules.py

Neural network building blocks for tabular models and diffusion-based generative models.

This module is largely adapted from:
    https://github.com/Yura52/rtdl

It provides:
- Activation functions (SiLU, ReGLU, GEGLU)
- Utility functions for timestep embeddings and activation handling
- Baseline MLP and ResNet architectures for tabular data
- Diffusion-specific network wrappers (MLPDiffusion, ResNetDiffusion)

Design notes
------------
- Architectures follow those described in Gorishniy et al. (2021) for tabular learning.
- Diffusion wrappers inject timestep embeddings (and optional label conditioning)
  before forwarding inputs through the base networks.
- This module defines **no script-level execution** and is safe to import anywhere.
"""

import math
from typing import Callable, List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) activation.

    Equivalent to `x * sigmoid(x)` and sometimes referred to as Swish.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    This is identical in spirit to positional encodings used in Transformers
    and is commonly used in diffusion models to encode time steps.

    Parameters
    ----------
    timesteps : torch.Tensor
        1D tensor of shape (N,) containing timestep indices (may be fractional).
    dim : int
        Dimensionality of the embedding.
    max_period : int, default=10000
        Controls the minimum frequency of the embeddings.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, dim) containing sinusoidal embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def _is_glu_activation(activation: ModuleType) -> bool:
    """Return True if the activation corresponds to a GLU-type activation."""
    return (
        isinstance(activation, str)
        and activation.endswith("GLU")
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    """Assert that all values are None or all are not None."""
    assert all(x is None for x in values) or all(x is not None for x in values)


def reglu(x: Tensor) -> Tensor:
    """
    ReGLU activation function.

    Splits the input in half along the last dimension and applies:
        a * ReLU(b)

    Reference
    ---------
    Shazeer, N., "GLU Variants Improve Transformer", 2020.
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """
    GEGLU activation function.

    Splits the input in half along the last dimension and applies:
        a * GELU(b)

    Reference
    ---------
    Shazeer, N., "GLU Variants Improve Transformer", 2020.
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """
    Module wrapper for the ReGLU activation.
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """
    Module wrapper for the GEGLU activation.
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    """
    Instantiate a PyTorch module from a string or callable specification.
    """
    if isinstance(module_type, str):
        if module_type == "ReGLU":
            return ReGLU()
        if module_type == "GEGLU":
            return GEGLU()
        return getattr(nn, module_type)(*args)
    return module_type(*args)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data.

    Architecture:
        (input) -> [Linear -> Activation -> Dropout] x N -> Linear -> (output)

    Based on:
        Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", 2021.
    """

    class Block(nn.Module):
        """Single MLP block: Linear → Activation → Dropout."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: ModuleType,
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type["MLP"],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> "MLP":
        """
        Construct the baseline MLP configuration used in the RTDL paper.
        """
        return cls(
            d_in=d_in,
            d_layers=d_layers,
            dropouts=dropout,
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ResNet(nn.Module):
    """
    Residual Network for tabular data.

    Architecture and defaults follow Gorishniy et al. (2021).
    """

    class Block(nn.Module):
        """Residual block with normalization, two linear layers and skip connection."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            residual = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            return residual + x if self.skip_connection else x

    class Head(nn.Module):
        """Final normalization → activation → linear head."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = self.normalization(x)
            x = self.activation(x)
            return self.linear(x)

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        super().__init__()
        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls: Type["ResNet"],
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> "ResNet":
        """
        Construct the baseline ResNet configuration from Gorishniy et al. (2021).
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization="BatchNorm1d",
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.first_layer(x)
        x = self.blocks(x)
        return self.head(x)


class MLPDiffusion(nn.Module):
    """
    Diffusion-ready MLP with timestep and optional label conditioning.
    """

    def __init__(self, d_in, num_classes, is_y_cond, rtdl_params, dim_t: int = 128):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        rtdl_params["d_in"] = dim_t
        rtdl_params["d_out"] = d_in
        self.mlp = MLP.make_baseline(**rtdl_params)

        if is_y_cond:
            self.label_emb = (
                nn.Embedding(num_classes, dim_t)
                if num_classes > 0
                else nn.Linear(1, dim_t)
            )

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

    def forward(self, x: Tensor, timesteps: Tensor, y: Tensor | None = None) -> Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            y = y.squeeze() if self.num_classes > 0 else y.view(y.size(0), 1).float()
            emb = emb + F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)


class ResNetDiffusion(nn.Module):
    """
    Diffusion-ready ResNet with timestep and optional label conditioning.
    """

    def __init__(self, d_in, num_classes, is_y_cond, rtdl_params, dim_t: int = 256):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes

        rtdl_params["d_in"] = d_in
        rtdl_params["d_out"] = d_in
        rtdl_params["emb_d"] = dim_t
        self.resnet = ResNet.make_baseline(**rtdl_params)

        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, dim_t)

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

    def forward(self, x: Tensor, timesteps: Tensor, y: Tensor | None = None) -> Tensor:
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if y is not None and self.num_classes > 0:
            emb = emb + self.label_emb(y.squeeze())
        return self.resnet(x, emb)
