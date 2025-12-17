"""
train.py

Prepare tabular data (periodic encoding + quantile normalization) and train a
Diffusion Probabilistic Model (DDPM) on real intervention data.

Pipeline overview
-----------------
1. Load `df_real.pkl` from `./Data_preprocessed`
2. Encode temporal variables (Month, Day, Hour) using sine/cosine features
3. Apply quantile normalization (to a normal distribution) on continuous variables
4. Serialize intermediate artifacts:
   - df_real.pkl (enriched with sin/cos features)
   - df_quantile.pkl (quantile-normalized data)
   - normalizer_ddpm.pkl (fitted QuantileTransformer)
   - dataset.pkl (training dataset)
5. Train a DDPM model using command-line hyperparameters

Notes
-----
- This script relies on multiple `os.chdir(...)` calls and therefore assumes a
  specific project directory structure.
- The "Incident" column is shifted to zero-based indexing (Incident -= 1),
  which is required by the downstream DDPM / tabular pipeline.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from data import *
from ddpm import DDPM


def quantile_transform(
    df_sincos: pd.DataFrame,
) -> tuple[pd.DataFrame, QuantileTransformer]:
    """
    Apply a quantile transformation (to a normal distribution) to continuous variables.

    This transformation is commonly used to stabilize training of tabular generative
    models (including DDPMs) by enforcing approximately Gaussian marginals.

    Transformed columns (hard-coded):
        - "Coord X"
        - "Coord Y"
        - "Duration"

    Parameters
    ----------
    df_sincos : pd.DataFrame
        Input DataFrame containing the continuous columns to transform.

    Returns
    -------
    df_transformed : pd.DataFrame
        The same DataFrame with transformed continuous columns.
    normalizer_ddpm : QuantileTransformer
        The fitted QuantileTransformer instance.

    Side Effects
    ------------
    Modifies the specified columns of `df_sincos` in place.

    Raises
    ------
    KeyError
        If one of the expected columns is missing.
    ValueError
        If the quantile transformation fails (e.g., NaNs or invalid shapes).
    """
    cols = ["Coord X", "Coord Y", "Duration"]
    values = df_sincos[cols].values

    normalizer_ddpm = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=1000,
        subsample=1_000_000_000,
        random_state=42,
    )

    df_sincos.loc[:, cols] = normalizer_ddpm.fit_transform(values)

    return df_sincos, normalizer_ddpm


def encode_periodic(value: float, period: float) -> tuple[float, float]:
    """
    Encode a periodic scalar variable using sine and cosine components.

    This encoding avoids artificial discontinuities at period boundaries
    (e.g., hour 23 vs hour 0).

    Parameters
    ----------
    value : float
        Scalar value to encode (e.g., month, day, hour).
    period : float
        Period of the variable (e.g., 12 for months, 24 for hours).

    Returns
    -------
    sin_value : float
        Sine component of the encoding.
    cos_value : float
        Cosine component of the encoding.
    """
    angle = (2 * np.pi * value) / period
    return np.sin(angle), np.cos(angle)


def sincos_transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add sine/cosine encodings for Month, Day, and Hour variables.

    Expected input columns:
        - "Month"
        - "Day"
        - "Hour"

    Generated features:
        - Month_sin, Month_cos
        - Day_sin, Day_cos
        - Hour_sin, Hour_cos

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input DataFrame containing temporal columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the six sine/cosine features,
        in a fixed and consistent order.

    Side Effects
    ------------
    Adds new columns to `df_raw` before returning a filtered view.

    Raises
    ------
    KeyError
        If required temporal columns are missing.
    """
    df_raw[["Month_sin", "Month_cos"]] = df_raw["Month"].apply(
        lambda x: pd.Series(encode_periodic(x, 12))
    )
    df_raw[["Day_sin", "Day_cos"]] = df_raw["Day"].apply(
        lambda x: pd.Series(encode_periodic(x, 365))
    )
    df_raw[["Hour_sin", "Hour_cos"]] = df_raw["Hour"].apply(
        lambda x: pd.Series(encode_periodic(x, 24))
    )

    return df_raw[
        ["Month_sin", "Month_cos", "Hour_sin", "Hour_cos", "Day_sin", "Day_cos"]
    ]


if __name__ == "__main__":
    """
    Script entry point.

    Expected inputs
    ---------------
    ./Data_preprocessed/df_real.pkl
        Pickled DataFrame containing at least:
        ["Coord X", "Coord Y", "Duration", "Incident",
         "Month", "Day", "Hour", "Minute"]

    Generated outputs (./Data_trained/)
    -----------------------------------
    - df_real.pkl
        Enriched DataFrame with sine/cosine features.
    - df_quantile.pkl
        Quantile-normalized dataset.
    - normalizer_ddpm.pkl
        Fitted QuantileTransformer (pickle).
    - dataset.pkl
        Serialized dataset used for DDPM training.

    Training
    --------
    - Instantiates a `DDPM` model using CLI hyperparameters.
    - Triggers model training via `ddpm.fit(dataset)`.

    Notes
    -----
    - The script assumes execution from the project root.
    - The "Incident" variable is shifted to zero-based indexing.
    """

    os.chdir("./Data_preprocessed")

    df_real = pd.read_pickle("df_real.pkl")

    print(df_real.columns)

    os.chdir("../Data_trained/")

    df_sincos = sincos_transform(df_real.copy())

    df_real_temp = df_sincos.copy()
    df_real_temp[
        ["Coord X", "Coord Y", "Duration", "Incident", "Month", "Day", "Hour", "Minute"]
    ] = df_real[
        ["Coord X", "Coord Y", "Duration", "Incident", "Month", "Day", "Hour", "Minute"]
    ].copy()
    print(df_real_temp.columns)
    df_real_temp.to_pickle("df_real.pkl")

    df_sincos[["Coord X", "Coord Y", "Duration", "Incident"]] = df_real[
        ["Coord X", "Coord Y", "Duration", "Incident"]
    ].copy()

    print(df_sincos.columns)

    df_sincos["Incident"] -= 1  # for tabddpm, min incident should be 0 not 1

    df_quantile, normalizer_ddpm = quantile_transform(df_sincos)

    df_quantile.to_pickle("df_quantile.pkl")

    pickle.dump(normalizer_ddpm, open("normalizer_ddpm.pkl", "wb"))

    dataset = raw_dataset_from_df(df_quantile, [], dummy=False, col="Incident")

    pickle.dump(dataset, open("dataset.pkl", "wb"))

    parser = argparse.ArgumentParser(description="Train_params")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of time steps"
    )
    parser.add_argument("--layers", type=int, default=1024, help="Size of layers")
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument(
        "--dim_t", type=int, default=128, help="Timestep embedding dimensions"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument(
        "--model_name", type=str, default="mlp", help="Model type : mlp or resnet"
    )
    parser.add_argument(
        "--gaussian_loss_type",
        type=str,
        default="mse",
        help="Gaussian loss type : mse or kl",
    )
    parser.add_argument(
        "--multinomial_loss_type",
        type=str,
        default="vb_stochastic",
        help="Multinomial loss type : vb_stochastic or vb_all",
    )
    parser.add_argument(
        "--parametrization",
        type=str,
        default="x0",
        help="Parametrization : x0 or direct",
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="Scheduler : cosine or linear"
    )
    parser.add_argument("--is_y_cond", action="store_true", help="Is target to predict")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--save_as", type=str, default="dqn", help="Save model in file")
    parser.add_argument(
        "--load_as", type=str, default="dqn", help="Load model from file"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")

    args = parser.parse_args()

    os.chdir("../")

    ddpm = DDPM(
        lr=args.lr,
        layers=args.layers,
        num_timesteps=args.num_timesteps,
        model_name=args.model_name,
        dim_t=args.dim_t,
        gaussian_loss_type=args.gaussian_loss_type,
        multinomial_loss_type=args.multinomial_loss_type,
        parametrization=args.parametrization,
        scheduler=args.scheduler,
        is_y_cond=args.is_y_cond,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        log_every=100,
        verbose=args.verbose,
        epochs=args.epochs,
        device=args.device,
        save_as=args.save_as,
        load_as=args.load_as,
    )

    ddpm.fit(dataset)

    print("Model trained")
