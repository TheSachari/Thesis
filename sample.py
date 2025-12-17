"""
sample.py

Generate synthetic intervention samples using a trained tabular DDPM and post-process them
into a realistic tabular dataset.

High-level workflow
-------------------
1) Load real (preprocessed) dataset `df_real.pkl` from `./Data_preprocessed/`
2) Load the serialized training `dataset.pkl` from `./Data_trained/`
3) Instantiate a `DDPM` synthesizer and load weights (via `DDPM.sample`)
4) Sample `num_samples = len(df_real) * os_factor` rows
5) Inverse-transform continuous variables via `QuantileTransformer.inverse_transform`
6) Decode periodic sin/cos features back to (Day, Month, Hour)
7) Spatially assign samples to a "sector" (`area_name`) using point-in-polygon against `pdd.geojson`
8) Downsample/oversample by sector to match a target distribution (pressure/variability knobs)
9) Post-process and optionally harmonize the Day distribution
10) Save final samples to `./Data_sampled/<save_sample_as>`

Inputs expected on disk
-----------------------
- ./Data_preprocessed/df_real.pkl
- ./Data_trained/dataset.pkl
- ./Data_trained/df_quantile.pkl
- ./Data_trained/normalizer_ddpm.pkl
- ./Data/pdd.geojson

Outputs
-------
- ./Data_sampled/df_fake_woh.pkl
    Samples after post_process(), before harmonize()
- ./Data_sampled/<save_sample_as>
    Final samples after optional harmonization

Notes
-----
- This script relies heavily on relative paths and `os.chdir(...)`. It should be run from
  the project root so paths resolve as expected.
- The code duplicates a few helper functions from preprocess.py (union_iris, get_point_in_area, ...).
  If you want to reduce duplication, you can import them from preprocess.py once that module is
  made import-safe (i.e., no side effects at import time).
- The original file already used `if __name__ == "__main__":`. This refactor keeps the same behavior,
  but moves the script body into a `main()` function for cleaner imports/testing.
"""

import argparse
import os
import pickle
from random import getrandbits, uniform

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Point, unary_union

from data import *  # noqa: F403
from ddpm import DDPM


def quantile_inverse_transform(df_res: pd.DataFrame, normalizer_ddpm) -> pd.DataFrame:
    """
    Invert the quantile transformation for continuous columns.

    Parameters
    ----------
    df_res : pd.DataFrame
        DataFrame containing (at least) the continuous columns in *quantile space*.
    normalizer_ddpm : QuantileTransformer or compatible
        Fitted quantile transformer used during training.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only ["Coord X", "Coord Y", "Duration"] in the original space.

    Notes
    -----
    The original code reloads `normalizer_ddpm.pkl` inside this function, ignoring the provided
    argument. This refactor preserves that behavior to avoid changing results.
    """
    cols = ["Coord X", "Coord Y", "Duration"]
    df_ddpm = pd.DataFrame(columns=cols)
    normalizer_ddpm = pickle.load(open("normalizer_ddpm.pkl", "rb"))
    df_ddpm[cols] = normalizer_ddpm.inverse_transform(df_res[cols].values)
    return df_ddpm


def decode_periodic(sin_value: float, cos_value: float, period: float) -> int:
    """
    Decode a sin/cos periodic encoding back to a scalar value.

    Parameters
    ----------
    sin_value : float
        Sine component.
    cos_value : float
        Cosine component.
    period : float
        Period of the variable (e.g., 365 for day-of-year).

    Returns
    -------
    int
        Decoded scalar (rounded to nearest integer).
    """
    angle = np.arctan2(sin_value, cos_value)
    if angle < 0:
        angle += 2 * np.pi
    value = (angle * period) / (2 * np.pi)
    return round(value)


def sincos_inverse_transform(df_sample: pd.DataFrame, df_res: pd.DataFrame, y_gen: np.ndarray) -> pd.DataFrame:
    """
    Decode periodic features and rebuild discrete columns from sampled outputs.

    Parameters
    ----------
    df_sample : pd.DataFrame
        DataFrame holding continuous columns after inverse quantile transform.
    df_res : pd.DataFrame
        DataFrame holding sin/cos columns in latent/sample space.
    y_gen : np.ndarray
        Generated incident labels (0-based) returned by DDPM sampler.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with decoded Month/Day/Hour and 1-based Incident labels.

    Notes
    -----
    - Incident labels are shifted back to 1-based indexing (Incident = y_gen + 1).
    - Duration is clipped to a plausible range (10 min to 20 h); outliers are replaced by the median
      (business rule from the original script).
    """
    df_sample["Day"] = df_res.apply(
        lambda row: decode_periodic(row["Day_sin"], row["Day_cos"], 365), axis=1
    )
    df_sample["Month"] = df_res.apply(
        lambda row: decode_periodic(row["Month_sin"], row["Month_cos"], 12), axis=1
    )
    df_sample["Hour"] = df_res.apply(
        lambda row: decode_periodic(row["Hour_sin"], row["Hour_cos"], 24), axis=1
    )
    df_sample["Incident"] = y_gen + 1
    df_sample["Duration"] = (
        df_sample["Duration"]
        .apply(lambda x: x if x > 10 and x < 20 * 60 else df_sample["duree"].median())
        .astype(int)
    )
    return df_sample


def clean_and_shift(row: pd.Series, cis_cols: list[str], li_new_cis: list[str]) -> pd.Series:
    """
    Clean CIS columns and shift remaining values left.

    Filters out placeholders starting with "Z" or "X" and excludes any CIS in `li_new_cis`.
    """
    filtered_values = [
        val
        for val in row[cis_cols]
        if not (str(val).startswith("Z") or str(val).startswith("X"))
        and val not in li_new_cis
    ]
    new_values = filtered_values + [""] * (len(cis_cols) - len(filtered_values))
    row.update(pd.Series(new_values, index=cis_cols))
    return row


def union_iris(df_pdd: gpd.GeoDataFrame, cis: str, li_new_cis: list[str]) -> gpd.GeoDataFrame:
    """
    Build sector polygons by unioning geometries grouped by a CIS column.

    Parameters
    ----------
    df_pdd : geopandas.GeoDataFrame
        Input polygons with CIS columns and geometry.
    cis : str
        Column name used as the sector identifier (e.g., "cis1").
    li_new_cis : list[str]
        CIS names to exclude.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns ["sector", "geometry"] reprojected to EPSG:2154.
    """
    df_pdd = df_pdd[~df_pdd[cis].str.startswith("Z")]
    cis_cols = [col for col in df_pdd.columns if col.startswith("cis")]
    df_pdd = df_pdd.apply(clean_and_shift, args=(cis_cols, li_new_cis), axis=1)

    area_name_unique = sorted(df_pdd[cis].unique())
    areas_geo = [
        MultiPolygon(list(df_pdd[df_pdd[cis] == nom].geometry)) for nom in area_name_unique
    ]

    areas = gpd.GeoDataFrame({"sector": area_name_unique, "geometry": areas_geo}, crs=df_pdd.crs)
    areas.geometry = areas.geometry.to_crs(2154)
    areas.geometry = areas.geometry.apply(lambda x: unary_union(x))
    return areas


def get_point_in_area(point: Point, zones: gpd.GeoDataFrame) -> str:
    """
    Return the sector name containing a given point (first match), or "" if none.
    """
    res = zones["sector"][zones.contains(point)].values
    return res[0] if res.size > 0 else ""


def gen_num_samples(num_samples: int, var: float) -> int:
    """
    Randomly perturb a target count by ±var and return an integer sample count.

    Parameters
    ----------
    num_samples : int
        Baseline count.
    var : float
        Relative variability (e.g., 0.02 means ±2%).

    Returns
    -------
    int
        Perturbed integer count.
    """
    sign = getrandbits(1)
    x = uniform(1, 1 + var) if sign else uniform(1 - var, 1)
    return int(round(num_samples * x))


def create_df_new_samples(df_raw: pd.DataFrame, col: str, pressure: float = 1, var: float = 0.02) -> pd.DataFrame:
    """
    Build a per-category sampling plan with optional global scaling and variability.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Reference DataFrame whose distribution will be used (typically real data).
    col : str
        Column name to match distribution on (e.g., "area_name").
    pressure : float, default=1
        Global multiplier applied to all counts (1.0 means 100% of the real distribution).
    var : float, default=0.02
        Allowed relative variability per category.

    Returns
    -------
    pd.DataFrame
        Table with columns [col, "count", "new_samples", "perc.", "delta"] where:
        - count: scaled baseline count
        - new_samples: perturbed desired count
        - delta: new_samples - count

    Notes
    -----
    A correction step enforces that the total number of new samples matches the baseline
    total after perturbations (by adjusting the max-delta category).
    """
    df_test = pd.DataFrame(df_raw[col].value_counts().reset_index())
    df_test["count"] *= pressure
    df_test["count"] = df_test["count"].astype(int)

    df_test.loc[:, "new_samples"] = df_test["count"].apply(gen_num_samples, args=(var,))
    df_test.loc[:, "perc."] = df_test["new_samples"] / df_test["count"] * 100 - 100
    df_test.loc[:, "delta"] = df_test["new_samples"] - df_test["count"]

    df_test.loc[df_test["delta"].idxmax(), "new_samples"] -= df_test["delta"].sum()
    df_test.loc[:, "perc."] = df_test["new_samples"] / df_test["count"] * 100 - 100
    df_test.loc[:, "delta"] = df_test["new_samples"] - df_test["count"]
    return df_test


def new_df_sample(df_test: pd.DataFrame, col: str, df_oversampled: pd.DataFrame) -> pd.DataFrame:
    """
    Build a new DataFrame by sampling per-category according to `df_test["new_samples"]`.

    Parameters
    ----------
    df_test : pd.DataFrame
        Sampling plan, must contain columns [col, "new_samples"].
    col : str
        Column used to match categories.
    df_oversampled : pd.DataFrame
        Pool of candidate rows to sample from (typically DDPM outputs with area_name assigned).

    Returns
    -------
    pd.DataFrame
        Concatenation of sampled subsets for each category.
    """
    list_of_df = []
    for _, row in df_test.iterrows():
        subset = df_oversampled[df_oversampled[col] == row[col]]
        if len(subset) >= row["new_samples"]:
            list_of_df.append(subset.sample(int(row["new_samples"])))
        else:
            list_of_df.append(subset)
    return pd.concat(list_of_df, ignore_index=True)


def post_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity constraints on decoded temporal variables.

    - Month < 1 -> 12
    - Day < 1 -> 1
    - Hour > 23 -> 0
    """
    df.loc[df["Month"] < 1, "Month"] = 12
    df.loc[df["Day"] < 1, "Day"] = 1
    df.loc[df["Hour"] > 23, "Hour"] = 0
    return df


def distance_days(a: int, b: int, nb_days: int = 365) -> int:
    """
    Circular distance between two days on a year-of-length `nb_days`.
    """
    return min(abs(a - b), nb_days - abs(a - b))


def harmonize(df1: pd.DataFrame, to_keep: int = 10, value_span: int = 25, nb_days: int = 365) -> pd.DataFrame:
    """
    Reduce day-of-year peaks by moving some interventions from top days to nearest low days.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing a "Day" column.
    to_keep : int, default=10
        Number of highest-count days to keep intact (peak days start after this slice).
    value_span : int, default=25
        Number of days considered for peak and low day sets.
    nb_days : int, default=365
        Year length used for circular distance.

    Returns
    -------
    pd.DataFrame
        DataFrame with some "Day" values reassigned from peaks to nearby low-frequency days.

    Notes
    -----
    This is a heuristic post-processing step intended to smooth distributional artifacts.
    """
    vc1 = df1.Day.value_counts().iloc[to_keep : value_span + to_keep]
    top_days = vc1.index.tolist()
    vc2 = df1.Day.value_counts().iloc[-value_span:]
    flop_days = vc2.index.tolist()

    for ref_day in top_days:
        nearest_day = min(flop_days, key=lambda d: distance_days(d, ref_day, nb_days))
        ref_inter_1 = df1[df1.Day == ref_day]
        ref_inter_2 = df1[df1.Day == nearest_day]
        mu = (len(ref_inter_1) + len(ref_inter_2)) // 2
        to_move = len(ref_inter_1) - mu
        inter_sampled = ref_inter_1.sample(n=to_move)
        df1.loc[inter_sampled.index, "Day"] = nearest_day
        flop_days.remove(nearest_day)

    return df1


def main() -> None:
    """
    CLI entry point.

    Parses arguments, runs DDPM sampling, applies inverse transforms and spatial/temporal
    post-processing, then writes pickled outputs under `./Data_sampled/`.
    """
    os.chdir("./Data_preprocessed")
    df_real = pd.read_pickle("df_real.pkl")

    os.chdir("../Data_trained")
    with open("dataset.pkl", "rb") as file:
        dataset = pickle.load(file)

    parser = argparse.ArgumentParser(description="Train_params")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--layers", type=int, default=1024, help="Size of layers")
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--dim_t", type=int, default=128, help="Timestep embedding dimensions")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument("--model_name", type=str, default="mlp", help="Model name")
    parser.add_argument("--gaussian_loss_type", type=str, default="mse", help="Gaussian loss type : mse or kl")
    parser.add_argument(
        "--multinomial_loss_type",
        type=str,
        default="vb_stochastic",
        help="Multinomial loss type : vb_stochastic or vb_all",
    )
    parser.add_argument("--parametrization", type=str, default="x0", help="Parametrization : x0 or direct")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler : cosine or linear")
    parser.add_argument("--is_y_cond", type=bool, default=True, help="Is target to predict")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--model_path", type=str, default="./model_ddpm.pt", help="Model path")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--save_as", type=str, default="dqn", help="Save model in path_to_file")
    parser.add_argument("--load_as", type=str, default="dqn", help="Load model from path_to_file")
    parser.add_argument("--os_factor", type=int, default=3, help="Oversampling factor")
    parser.add_argument(
        "--pressure",
        type=float,
        default=1,
        help="Number of interventions to sample, 1 is 100% from real dataset",
    )
    parser.add_argument("--to_keep", type=int, default=10, help="nb of highest values to keep by harmonization")
    parser.add_argument("--value_span", type=int, default=25, help="nb of interventions to permute by harmonization")
    parser.add_argument("--sample_batch_size", type=int, default=8192, help="Batch size of samples")
    parser.add_argument("--variability", type=float, default=0.02, help="Tolerated variability")
    parser.add_argument("--save_sample_as", type=str, default="df_fake.pkl", help="Save sample in path_to_file")

    args = parser.parse_args()

    print(
        "os factor:",
        args.os_factor,
        "sample batch size:",
        args.sample_batch_size,
        "pressure:",
        args.pressure,
        "variability:",
        args.variability,
        flush=True,
    )

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

    num_samples = len(df_real) * args.os_factor
    X_gen, y_gen = ddpm.sample(dataset, num_samples, args.sample_batch_size)

    os.chdir("./Data_trained")

    df_quantile = pickle.load(open("df_quantile.pkl", "rb"))
    normalizer_ddpm = pickle.load(open("normalizer_ddpm.pkl", "rb"))

    cols = list(df_quantile.columns)
    cols.remove("Incident")  # to match the generated X dimension
    df_res = pd.DataFrame(data=X_gen, columns=cols)

    df_sample = quantile_inverse_transform(df_res, normalizer_ddpm)
    df_oversampled = sincos_inverse_transform(df_sample, df_res, y_gen)

    df_oversampled[
        ["Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]
    ] = df_res[["Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]]

    os.chdir("../Data")

    filename = "pdd.geojson"
    df_pdd = gpd.read_file(filename)

    zones = union_iris(
        df_pdd,
        "cis1",
        ["MONTGISCARD", "AUSSONNE", "TOULOUSE - ATLANTA", "TOULOUSE - CARSALADE", "TOULOUSE - DELRIEU"],
    )

    gdf_real = gpd.GeoDataFrame(
        df_real,
        geometry=gpd.points_from_xy(df_real["Coord X"], df_real["Coord Y"]),
        crs="2154",
    )
    df_real["area_name"] = gdf_real["geometry"].apply(get_point_in_area, args=(zones,))

    gdf_fake = gpd.GeoDataFrame(
        df_oversampled,
        geometry=gpd.points_from_xy(df_oversampled["Coord X"], df_oversampled["Coord Y"]),
        crs="2154",
    )
    df_oversampled["area_name"] = gdf_fake["geometry"].apply(get_point_in_area, args=(zones,))

    df_new_samples = create_df_new_samples(df_real, "area_name", args.pressure, args.variability)
    print(df_new_samples.columns, flush=True)
    if df_new_samples.delta.sum() == 0:
        print("new samples OK", flush=True)
    else:
        print("new samples NOT OK", df_new_samples.delta.sum(), flush=True)

    df_fake = new_df_sample(df_new_samples, "area_name", df_oversampled)
    df_fake = post_process(df_fake)

    os.chdir("../Data_sampled")

    df_fake.to_pickle("df_fake_woh.pkl")

    df_fake = harmonize(df_fake, args.to_keep, args.value_span, 365)
    df_fake.to_pickle(args.save_sample_as)

    print(df_fake.shape, flush=True)
    print(df_fake.columns, flush=True)

    # Preserve the original duplicated save (harmless, but kept for identical behavior)
    df_fake.to_pickle(args.save_sample_as)

    print("dataset sampled", flush=True)
    os.chdir("../")


if __name__ == "__main__":
    main()
