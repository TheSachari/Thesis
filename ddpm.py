"""
ddpm.py

Training and sampling wrapper for a tabular DDPM (Denoising Diffusion Probabilistic Model)
with mixed numerical + categorical data.

This module defines:
- `RandomWalkDataset`: an IterableDataset that samples uniformly from a dataset
- `to_good_ohe`: helper to "harden" one-hot encoded blocks (turns soft/probabilistic OHE into {0,1})
- `DDPM`: a `BaseSynthesizer` implementation providing `fit()` and `sample()`

Key dependencies (project-specific)
----------------------------------
- `GaussianMultinomialDiffusion` implements the diffusion process for mixed data types.
- `get_model` builds the denoising network (MLP/ResNet depending on `model_name`).
- `prepare_fast_dataloader` yields fast mini-batches from the tabular `Dataset` container.
- `schedulefree.AdamWScheduleFree` is used as the optimizer (no LR schedule by default),
  plus an additional cosine warm restart scheduler is instantiated (currently unused).

No side effects
---------------
This file contains no top-level script logic (no I/O, no training triggered at import time).
It is safe to import from other modules and notebooks.
"""

import random
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import schedulefree
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from base import BaseSynthesizer, random_state
from data import prepare_fast_dataloader
from gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from utils_train import get_model, update_ema

# Project modules (likely define model blocks, layers, etc.)
from modules import *  # noqa: F403, F401


class RandomWalkDataset(IterableDataset):
    """
    An infinite iterable dataset that yields random samples from an indexable dataset.

    This is useful when you want a stream of samples without epoch boundaries,
    especially in training loops that do not rely on finite-length DataLoaders.

    Parameters
    ----------
    dataset : Sequence-like
        Any indexable object supporting `__len__` and `__getitem__`.

    Notes
    -----
    - Sampling is uniform over indices [0, len(dataset)-1].
    - This dataset yields single items (not batches).
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.size = len(dataset)

    def __iter__(self):
        while True:
            yield self.dataset[random.randint(0, self.size - 1)]


def to_good_ohe(ohe, X: np.ndarray) -> np.ndarray:
    """
    Convert (potentially soft) one-hot encoded blocks into hard {0, 1} one-hot vectors.

    Some pipelines produce one-hot-like blocks where values are not strictly 0/1
    (e.g., logits, probabilities, or numerically unstable encodings). This helper:
    - splits X into blocks using `ohe._n_features_outs`
    - finds the maximum per row within each block
    - sets entries equal to the row-wise max to 1, all others to 0

    Parameters
    ----------
    ohe : object
        An encoder-like object exposing `_n_features_outs`, a list of output sizes per
        original categorical feature.
    X : np.ndarray, shape (N, sum(ohe._n_features_outs))
        Concatenated one-hot blocks.

    Returns
    -------
    np.ndarray
        Hardened one-hot matrix with the same shape as X, containing only 0/1 values.

    Notes
    -----
    - Ties are allowed: if multiple entries share the maximum, they will all become 1.
    - This function assumes `X` contains only concatenated OHE blocks (no numerical features).
    """
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1] : indices[i]], axis=1)
        t = X[:, indices[i - 1] : indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


class DDPM(BaseSynthesizer):
    """
    Tabular DDPM synthesizer for mixed numerical + categorical data.

    This class wraps model construction, training, EMA tracking, checkpoint saving,
    and sampling for a diffusion model implemented in `GaussianMultinomialDiffusion`.

    Parameters
    ----------
    lr : float
        Learning rate.
    layers : int
        Hidden layer width used in the denoising network configuration.
    num_timesteps : int
        Number of diffusion timesteps.
    model_name : str, default="mlp"
        Denoiser architecture name (as expected by `get_model`), e.g. "mlp" or "resnet".
    dim_t : int, default=128
        Timestep embedding dimension.
    gaussian_loss_type : str, default="mse"
        Loss for numerical diffusion component (e.g. "mse" or "kl").
    multinomial_loss_type : str, default="vb_all"
        Loss for categorical diffusion component (e.g. "vb_all" or "vb_stochastic").
    parametrization : str, default="x0"
        Diffusion parametrization (e.g. "x0" or "direct").
    scheduler : str, default="cosine"
        Diffusion scheduler type (not the optimizer scheduler).
    is_y_cond : bool, default=False
        Whether to condition the denoiser on targets `y`.
    weight_decay : float, default=0
        Weight decay for the optimizer.
    batch_size : int, default=500
        Batch size. Must be even (asserted).
    log_every : int, default=100
        Frequency (in steps) at which losses are aggregated for logging.
    verbose : bool, default=False
        If True, enables tqdm descriptions.
    epochs : int, default=300
        Number of optimization steps (named epochs in the original code, but used as steps).
    device : str, default="cuda"
        Preferred device (CUDA if available, else CPU).
    save_as : str, default="_dqn_"
        Name suffix for saving model checkpoints.
    load_as : str, default="_dqn_"
        Name suffix for loading model checkpoints during sampling.

    Attributes
    ----------
    diffusion : GaussianMultinomialDiffusion
        The diffusion module (created in `fit` and `sample`).
    ema_model : torch.nn.Module
        EMA copy of the denoising model weights (created in `fit`).
    optimizer : torch.optim.Optimizer
        Optimizer instance (ScheduleFree AdamW variant).
    """

    def __init__(
        self,
        lr: float,
        layers: int,
        num_timesteps: int,
        model_name: str = "mlp",
        dim_t: int = 128,
        gaussian_loss_type: str = "mse",
        multinomial_loss_type: str = "vb_all",
        parametrization: str = "x0",
        scheduler: str = "cosine",
        is_y_cond: bool = False,
        weight_decay: float = 0,
        batch_size: int = 500,
        log_every: int = 100,
        verbose: bool = False,
        epochs: int = 300,
        device: str = "cuda",
        save_as: str = "_dqn_",
        load_as: str = "_dqn_",
    ):
        assert batch_size % 2 == 0

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.disbalance = None
        self.steps = epochs
        self.layers = layers
        self.init_lr = lr
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.multinomial_loss_type = multinomial_loss_type
        self.parametrization = parametrization
        self.scheduler = scheduler

        self.is_y_cond = is_y_cond
        self.dim_t = dim_t
        self.model_name = model_name

        self._transformer = None
        self._data_sampler = None

        self.ema_every = 1000

        # Use CUDA if available, otherwise fall back to CPU (ignores the `device` string if CUDA not available).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_as = save_as
        self.load_as = load_as
        self._verbose = verbose
        self.loss_values = None

    def _anneal_lr(self, step: int) -> None:
        """
        Linearly anneal the learning rate from `init_lr` to 0 over `self.steps`.

        Parameters
        ----------
        step : int
            Current optimization step (0-indexed).

        Notes
        -----
        This method is currently not called in the training loop.
        """
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x: torch.Tensor, out_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one optimization step and return component losses.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of concatenated features (numerical + categorical encodings),
            shape (B, D).
        out_dict : dict[str, torch.Tensor]
            Conditioning dictionary passed to the diffusion model.
            In this codebase, it is typically {"y": y_batch}.

        Returns
        -------
        loss_multi : torch.Tensor
            Categorical / multinomial loss component.
        loss_gauss : torch.Tensor
            Numerical / Gaussian loss component.
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)

        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    @random_state
    def fit(self, dataset) -> None:
        """
        Train the diffusion model on the provided tabular dataset.

        Parameters
        ----------
        dataset : Dataset
            A tabular dataset object (see `data.py`) exposing:
            - `X_num["train"]` (optional)
            - `X_cat["train"]` (optional)
            - `y["train"]`
            - `get_category_sizes("train")`
            - `n_classes`

        Side Effects
        ------------
        - Creates and stores `self.diffusion`, `self.ema_model`, and `self.optimizer`.
        - Writes checkpoint files into `./SVG_model/`:
          - model_<save_as>_diffusion.pt
          - model_<save_as>_ema.pt

        Notes
        -----
        - `self.steps` controls the number of optimization iterations.
        - Targets are wrapped into out_dict as {"y": y_batch}.
        - A cosine warm restart scheduler is instantiated but not stepped in the loop.
        """
        K = np.array(dataset.get_category_sizes("train"))
        if len(K) == 0:
            K = np.array([0])
        print("K", K)

        num_numerical_features = (
            dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
        )
        d_in = int(np.sum(K) + num_numerical_features)
        print("features", d_in)

        model_params = {
            "d_in": d_in,
            "is_y_cond": self.is_y_cond,
            "num_classes": dataset.n_classes,
            "rtdl_params": {"d_layers": [self.layers, self.layers], "dropout": 0.0},
            "dim_t": self.dim_t,
        }
        print(model_params)

        denoiser = get_model(
            self.model_name,
            model_params,
            num_numerical_features,
            category_sizes=dataset.get_category_sizes("train"),
        )
        denoiser.to(self.device)

        self.train_iter = prepare_fast_dataloader(
            dataset, split="train", batch_size=self.batch_size
        )

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=denoiser,
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            multinomial_loss_type=self.multinomial_loss_type,
            parametrization=self.parametrization,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.diffusion.to(self.device)
        self.diffusion.train()
        print("diffusion ready")

        # Exponential Moving Average (EMA) model for more stable sampling.
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.optimizer = schedulefree.AdamWScheduleFree(
            self.diffusion.parameters(), lr=self.lr
        )
        self.optimizer.train()

        # NOTE: Currently instantiated but not used in the loop (no scheduler.step()).
        _scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=4000,
            T_mult=1,
            eta_min=1e-6,
        )

        step_iterator = tqdm(range(self.steps), disable=(not self._verbose))
        if self._verbose:
            description = "mloss ({mloss:.2f}) | gloss ({gloss:.2f})"
            step_iterator.set_description(description.format(mloss=0, gloss=0))

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_count = 0
        mloss, gloss = 0.0, 0.0

        for step in step_iterator:
            x, y = next(self.train_iter)
            out_dict = {"y": y}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(
                self.ema_model.parameters(), self.diffusion._denoise_fn.parameters()
            )

            if self._verbose:
                step_iterator.set_description(
                    description.format(mloss=mloss, gloss=gloss)
                )

        torch.save(
            self.diffusion._denoise_fn.state_dict(),
            "./SVG_model/model_" + self.save_as + "_diffusion.pt",
        )
        torch.save(
            self.ema_model.state_dict(), "./SVG_model/model_" + self.save_as + "_ema.pt"
        )

    @random_state
    def sample(self, dataset, num_samples: int = 0, batch_size: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples from a trained diffusion model.

        Parameters
        ----------
        dataset : Dataset
            Dataset object providing category sizes and training targets distribution.
        num_samples : int, default=0
            Number of samples to generate. If 0, defaults to the number of training rows.
        batch_size : int, default=2000
            Batch size used during sampling.

        Returns
        -------
        X_gen : np.ndarray
            Generated feature matrix (concatenated numerical + categorical encodings).
        y_gen : np.ndarray
            Generated target labels (if the model is y-conditional).

        Notes
        -----
        - This method loads the denoiser checkpoint:
          `./SVG_model/model_<load_as>_diffusion.pt`.
        - Sampling uses the empirical class distribution from `dataset.y["train"]`.
        """
        if num_samples == 0:
            num_samples, num_numerical_features = dataset.X_num["train"].shape
        else:
            num_numerical_features = dataset.X_num["train"].shape[1]

        self.batch_size = batch_size

        K = np.array(dataset.get_category_sizes("train"))
        if len(K) == 0:
            K = np.array([0])
        print(K)

        num_numerical_features_ = (
            dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
        )
        d_in = int(np.sum(K) + num_numerical_features_)
        print("features", d_in)

        model_params = {
            "d_in": d_in,
            "is_y_cond": self.is_y_cond,
            "num_classes": dataset.n_classes,
            "rtdl_params": {"d_layers": [self.layers, self.layers], "dropout": 0.0},
            "dim_t": self.dim_t,
        }
        print(model_params)

        denoiser = get_model(
            self.model_name,
            model_params,
            num_numerical_features_,
            category_sizes=dataset.get_category_sizes("train"),
        )

        denoiser.load_state_dict(
            torch.load(
                "./SVG_model/model_" + self.load_as + "_diffusion.pt",
                weights_only=True,
            )
        )
        print("params loaded")

        denoiser.to(self.device)

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features_,
            denoise_fn=denoiser,
            gaussian_loss_type=self.gaussian_loss_type,
            multinomial_loss_type=self.multinomial_loss_type,
            parametrization=self.parametrization,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.diffusion.eval()
        print("diffusion ready")

        _, empirical_class_dist = torch.unique(
            torch.from_numpy(dataset.y["train"]), return_counts=True
        )

        x_gen, y_gen = self.diffusion.sample_all(
            num_samples, batch_size, empirical_class_dist.float(), ddim=False
        )

        return x_gen.numpy(), y_gen.numpy()
