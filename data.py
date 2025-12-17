"""
data.py

Lightweight utilities for representing tabular datasets and building fast PyTorch
data loaders.

This module provides:
- `TaskType`: an enum describing the learning task (binary/multiclass/regression)
- `Dataset`: a simple container for numerical/categorical features and targets
- `raw_dataset_from_df`: helper to convert a pandas DataFrame into a `Dataset`
- `prepare_fast_dataloader`: an infinite generator yielding mini-batches
- `FastTensorDataLoader`: a fast tensor-only iterator (avoids per-index overhead)

Design notes
------------
- The `Dataset` class is intentionally minimal and stores arrays by split name
  (e.g., {"train": ...}). In this codebase only the "train" split is used.
- `prepare_fast_dataloader` concatenates numerical and categorical features into a
  single float tensor (after casting). This assumes categorical columns have been
  encoded numerically upstream. If you keep categories as strings, you must encode
  them before training.

No side effects
---------------
This file contains no top-level execution logic (no I/O, no directory changes).
It is safe to import from other modules and notebooks.
"""

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

ArrayDict = Dict[str, np.ndarray]


class TaskType(enum.Enum):
    """
    Enum describing the supervised learning task associated with a dataset.

    Values
    ------
    BINCLASS
        Binary classification.
    MULTICLASS
        Multiclass classification.
    REGRESSION
        Regression.
    """

    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        """Return the string value of the enum (useful for logging/CLI)."""
        return self.value


def raw_dataset_from_df(
    df,
    cat_features: Sequence[str],
    dummy: bool = True,
    col: str = "Incident",
):
    """
    Build a `Dataset` object from a pandas DataFrame.

    The function splits features into:
    - Numerical features: all columns except `cat_features` and (optionally) the target
    - Categorical features: `cat_features` (stored as strings when present)
    - Targets: either a dummy constant target (when `dummy=True`) or `df[col]`

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing features and (optionally) a target column.
    cat_features : Sequence[str]
        Names of categorical feature columns. If empty, `X_cat` will be set to None.
    dummy : bool, default=True
        If True, creates a dummy target vector of zeros and sets `task_type` to None.
        This is useful when training an unconditional generative model.
        If False, uses `df[col]` as targets and sets `task_type` to MULTICLASS.
    col : str, default="Incident"
        Target column name used when `dummy=False`.

    Returns
    -------
    Dataset
        A Dataset instance holding `X_num`, `X_cat` (optional), `y`, and metadata.

    Notes
    -----
    - When `dummy=False`, the task is hard-coded to MULTICLASS and `n_classes` is
      computed from unique target values.
    - Categorical features are converted to `str` here. If your model expects numeric
      encodings, convert/encode categories before calling `prepare_fast_dataloader`.
    """
    y: ArrayDict = {}
    X_num: ArrayDict = {}
    X_cat: Optional[ArrayDict] = {} if len(cat_features) else None
    target_cols: List[str] = []

    if dummy:
        y["train"] = np.array([0.0] * len(df))
        task_type = None
        n_classes = 0
    else:
        y["train"] = df[col].to_numpy()
        task_type = TaskType("multiclass")
        n_classes = len(np.unique(y["train"]))
        target_cols = [col]

    if X_cat is not None:
        X_cat["train"] = df[list(cat_features)].to_numpy().astype(str)

    X_num["train"] = (
        df.drop(list(cat_features) + target_cols, axis=1).to_numpy().astype(float)
    )

    y_info = {"policy": "default"}

    dataset = Dataset(X_num, X_cat, y, y_info, task_type, n_classes)
    return dataset


@dataclass(frozen=False)
class Dataset:
    """
    Minimal container for tabular datasets split by partition name.

    Attributes
    ----------
    X_num : Optional[Dict[str, np.ndarray]]
        Numerical features per split, e.g. {"train": (N, D_num)}.
    X_cat : Optional[Dict[str, np.ndarray]]
        Categorical features per split, e.g. {"train": (N, D_cat)}. May be None.
    y : Dict[str, np.ndarray]
        Targets per split, e.g. {"train": (N,)}.
    y_info : Dict[str, Any]
        Additional metadata about the targets.
    task_type : TaskType
        Task type (BINCLASS, MULTICLASS, REGRESSION). May be None if `dummy=True`.
    n_classes : Optional[int]
        Number of classes for classification tasks.
    """

    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @property
    def is_binclass(self) -> bool:
        """Whether the dataset is binary classification."""
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        """Whether the dataset is multiclass classification."""
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        """Whether the dataset is regression."""
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        """Number of numerical features (0 if X_num is None)."""
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        """Number of categorical features (0 if X_cat is None)."""
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        """Total number of features (numerical + categorical)."""
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        """
        Return the number of samples.

        Parameters
        ----------
        part : Optional[str]
            Split name ("train", "val", "test") or None to sum over all splits.

        Returns
        -------
        int
            Number of samples.
        """
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        """
        Output dimension expected from a neural network head.

        Returns
        -------
        int
            `n_classes` for multiclass tasks, otherwise 1.
        """
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        return 1

    def get_category_sizes(self, part: str) -> List[int]:
        """
        Return category cardinalities for each categorical feature.

        Parameters
        ----------
        part : str
            Split name to compute category sizes for (typically "train").

        Returns
        -------
        List[int]
            Category sizes. Returns an empty list if `X_cat` is None.

        Notes
        -----
        This calls `get_category_sizes` from elsewhere in the codebase (if available).
        """
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])


def prepare_fast_dataloader(D: Dataset, split: str, batch_size: int):
    """
    Create an *infinite* generator yielding mini-batches of (X, y).

    This utility concatenates numerical and categorical features (if present) into
    a single tensor and uses `FastTensorDataLoader` for fast iteration.

    Parameters
    ----------
    D : Dataset
        Dataset object containing features and targets.
    split : str
        Split name to iterate over (e.g., "train").
    batch_size : int
        Mini-batch size.

    Yields
    ------
    (torch.Tensor, torch.Tensor)
        A tuple (X, y) where:
        - X has shape (B, D) and dtype float32
        - y has shape (B,) (dtype depends on `D.y[split]`)

    Notes
    -----
    - This generator never terminates. It is intended for training loops that run
      for a fixed number of steps/epochs.
    - If `D.X_cat` contains non-numeric strings, `torch.from_numpy(...).float()` will fail.
      Encode categories to numeric values before calling this function.
    """
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()

    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split == "train"))

    while True:
        yield from dataloader


class FastTensorDataLoader:
    """
    A lightweight, fast DataLoader-like iterator over one or more tensors.

    This avoids the overhead of `TensorDataset + DataLoader`, which can be slower
    because the DataLoader fetches individual indices and concatenates them.

    Source:
        https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6

    Parameters
    ----------
    *tensors : torch.Tensor
        Tensors to iterate over. All must share the same length along dimension 0.
    batch_size : int, default=32
        Mini-batch size.
    shuffle : bool, default=False
        Whether to shuffle tensors in-place at each new iterator creation.

    Notes
    -----
    - Shuffling is performed by permuting dim-0 indices and reindexing the tensors.
    - This class yields a tuple of tensors (one batch slice per input tensor).
    """

    def __init__(self, *tensors, batch_size: int = 32, shuffle: bool = False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = list(tensors)

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        self.n_batches = n_batches + (1 if remainder > 0 else 0)

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.n_batches
