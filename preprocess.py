"""
preprocess.py

Preprocessing utilities and runnable pipeline for emergency intervention data.

This module exposes helper functions (e.g., `preprocess`, `union_iris`) that can be
imported without triggering I/O. When run as a script, it loads raw files from
`./Data/`, runs preprocessing, and writes pickled artifacts to `./Data_preprocessed/`.
"""

import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Point, unary_union


def replace_vl_prefix(lst):
    return [re.sub(r"^VL\s+", "VL", item) for item in lst]


def replace_in_list(lst, dic):
    return [dic.get(item, item) for item in lst]


def get_point_in_area(point, zones):
    res = zones["sector"][zones.contains(point)].values
    if res.size > 0:
        return res[0]
    else:
        return ""


def get_centroid(sector, zones):
    if sector in zones["sector"].values:
        return zones["centroid"][zones["sector"] == sector].values[0]
    else:
        return Point(0, 0)


def preprocess(gdf_secteurs, df_sorties, df_inter, df_materiel):
    zones = union_iris(
        gdf_secteurs,
        "cis1",
        [
            "MONTGISCARD",
            "AUSSONNE",
            "TOULOUSE - ATLANTA",
            "TOULOUSE - CARSALADE",
            "TOULOUSE - DELRIEU",
        ],
    )
    zones["centroid"] = zones["geometry"].apply(lambda x: x.centroid)

    df_sort = df_sorties[
        (df_sorties["Nom du Centre"].isin(zones["sector"]))
        & (df_sorties["Ordre Renfort"] == "P")
        & (df_sorties["Annulation de départ (O/N)"] == "non")
    ]
    to_ban = [
        "ACTIVATION PCA",
        "DEPLACEMENT VEHICULE OPERATIONNEL SPORT",
        "ESSAI ",
        "MANOEUVRES ET EXERCICES",
        "MANOEUVRES ET EXERCICES SPV",
    ]
    df_inter = df_inter[(~df_inter["Sinistre initial - Nom"].isin(to_ban))]
    df_inter = pd.merge(
        df_sort, df_inter, on="Numéro d'intervention", how="inner"
    ).reset_index(drop=True)
    df_inter = df_inter[
        [
            "Numéro d'intervention",
            "Date Heure Début Intervention",
            "Date Heure fin Intervention",
            "Sinistre initial - Nom",
            "Localisation initiale - Coord X",
            "Localisation initiale - Coord Y",
        ]
    ]
    df_inter["coord_x"] = df_inter["Localisation initiale - Coord X"]
    df_inter["coord_y"] = df_inter["Localisation initiale - Coord Y"]
    df_inter = df_inter.dropna(subset=(["coord_x", "coord_y"])).reset_index(drop=True)

    frmt = "%d/%m/%Y %H:%M:%S"
    df_inter["heure_debut"] = df_inter["Date Heure Début Intervention"]
    df_inter["heure_debut"] = pd.to_datetime(df_inter["heure_debut"], format=frmt)
    s = pd.Series(df_inter["heure_debut"].values.astype(np.float32))
    s[s < 0] = np.nan
    df_inter["heure_debut"] = pd.to_datetime(s.interpolate())
    df_inter["heure_debut"] = df_inter["heure_debut"].dt.floor("s")

    df_inter["heure_fin"] = df_inter["Date Heure fin Intervention"]
    df_inter["heure_fin"] = pd.to_datetime(df_inter["heure_fin"], format=frmt)
    s = pd.Series(df_inter["heure_fin"].values.astype(np.float32))
    s[s < 0] = np.nan
    df_inter["heure_fin"] = pd.to_datetime(s.interpolate())

    df_inter["duree"] = df_inter["heure_fin"] - df_inter["heure_debut"]
    df_inter["duree"] = df_inter["duree"].dt.total_seconds().div(60).astype(int)
    df_inter["duree"] = (
        df_inter["duree"]
        .apply(lambda x: x if x > 10 and x < 20 * 60 else df_inter["duree"].median())
        .astype(float)
    )

    df_inter["heure"] = df_inter["heure_debut"].dt.hour
    df_inter["jour"] = df_inter["heure_debut"].dt.dayofyear
    df_inter["mois"] = df_inter["heure_debut"].dt.month
    df_inter["minute"] = df_inter["heure_debut"].dt.minute

    df_inter["Sinistre initial - Nom"] = df_inter["Sinistre initial - Nom"].str.replace(
        r"^ZZ_U2", "U1", regex=True
    )
    dic_replace = {
        k: v
        for k, v in zip(
            [
                "U1 DECLENCHEMENT TELEASSISTANCE AVEC SUSPICION DE DETRESS",
                "AUTRES FEUX EN PLEIN AIR (SANS PRÉCISION)",
                "DECLENCHEMENT DE DETECTEUR FUMEE OU CO",
                "MATERIAUX MENACANT RUINE =< R+2",
                "AUTRE FEU D'HABITATION (COMPTEUR EDF - ETC.)",
            ],
            [
                "U1 DECLENCHEMENT TELEASSISTANCE AVEC SUSPICION DE DETRESSE",
                "AUTRES FEUX (SANS PRÉCISION)",
                "DECLENCHEMENT DE DETECTEUR DE FUMEE",
                "MATERIAUX MENACANT RUINE",
                "AUTRE FEU D'HABITATION (COMPTEUR EDF....)",
            ],
        )
    }

    df_inter["Sinistre initial - Nom"] = df_inter["Sinistre initial - Nom"].replace(
        dic_replace
    )

    v = df_inter["Sinistre initial - Nom"].value_counts()
    df_inter = df_inter[
        df_inter["Sinistre initial - Nom"].isin(v.index[v.gt(100)])
    ].reset_index(drop=True)
    df_inter

    dic_sin_rank = {
        k: v
        for k, v in zip(
            df_inter["Sinistre initial - Nom"].value_counts().index,
            range(1, len(df_inter["Sinistre initial - Nom"].value_counts()) + 1),
        )
    }
    df_inter.loc[:, "sin_rank"] = (
        df_inter["Sinistre initial - Nom"].map(dic_sin_rank).fillna(0)
    )

    df_rank_incident = {v: k for k, v in dic_sin_rank.items()}
    df_rank_incident = pd.DataFrame(df_rank_incident.items(), columns=["rank", "sin"])

    gdf_real = gpd.GeoDataFrame(
        df_inter,
        geometry=gpd.points_from_xy(df_inter["coord_x"], df_inter["coord_y"]),
        crs="2154",
    )

    df_inter["sector"] = gdf_real["geometry"].apply(get_point_in_area, args=(zones,))
    df_inter["centroid"] = df_inter["sector"].apply(get_centroid, args=(zones,))

    df_inter["centroid_x"] = df_inter["centroid"].apply(lambda p: p.x)
    df_inter["centroid_y"] = df_inter["centroid"].apply(lambda p: p.y)
    df_inter = df_inter[df_inter.centroid_x != 0]

    k = [
        "coord_x",
        "coord_y",
        "mois",
        "jour",
        "heure",
        "minute",
        "duree",
        "sin_rank",
        "Numéro d'intervention",
    ]
    v = [
        "Coord X",
        "Coord Y",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Duration",
        "Incident",
        "num_inter",
    ]

    res = dict(map(lambda i, j: (i, j), k, v))
    df_inter = df_inter[k]
    df_inter = df_inter.rename(columns=res, errors="raise")

    df_materiel = df_materiel[
        [
            "Numéro d'intervention",
            "IU Materiel",
            "Type materiel",
            "Fonction materiel",
            "Numero Depart",
            "Nom du Centre",
        ]
    ]

    df_materiel = df_materiel.rename(
        columns={
            "Numéro d'intervention": "num_inter",
            "Type materiel": "material",
            "Fonction materiel": "function",
            "Numero Depart": "num_dep",
            "Nom du Centre": "station",
        }
    )

    df_func = df_materiel.groupby("num_inter")["function"].apply(list).reset_index()
    df_func = df_func[df_func["num_inter"].isin(df_inter.num_inter)].reset_index(
        drop=True
    )
    df_inter = df_inter[df_inter["num_inter"].isin(df_func.num_inter)].reset_index(
        drop=True
    )

    df_inter["real_func"] = df_func["function"].copy()

    df_inter["real_func"] = df_inter["real_func"].apply(replace_vl_prefix)

    dic_replace = {
        "CCI": "CCFM",
        "BLR": "BMS",
        "FPT SD": "FPT",
        "EPS": "EPC18",
        "EPA DEGRAD": "EPC18",
        "XCOMPL": "COMPL",
        "RIMP": "VGRIMP",
        "EMB": "BLS",
        "GRIMP": "VGRIMP",
        "VLOG": "VLA",
        "VPL": "VSN",
        "CCF DEGRAD": "CCFM",
    }

    df_inter["real_func"] = df_inter["real_func"].apply(
        lambda lst: replace_in_list(lst, dic_replace)
    )

    li = [
        "VL OFF RT",
        "CEPRO",
        "DA",
        "CEAEGC",
        "CCI",
        "BLR",
        "DA",
        "VL CODIS M",
        "FPT SD",
        "CESD",
        "EPS",
        "VL MOY PC",
        "VL RENS PC",
        "MPR",
        "CDHR",
        "EPA DEGRAD",
        "CEIN",
        "XCOMPL",
        "MPRG",
        "VL CODIS C",
        "CDHR",
        "CDEHR",
        "PT TRANSIT",
        "RIMP",
        "EMB",
        "CAMPLATEAU",
        "GRIMP",
        "CESDMF",
        "CEPOL",
        "CAMERA",
        "RPO",
        "VL MECANIQ",
        "VLMECANIQ",
        "VL MCHEF",
        "CESAP",
        "VPL",
        "VLOG",
        "VSAT",
        "CCF DEGRAD",
    ]
    df_inter["real_func"] = df_inter["real_func"].apply(
        lambda lst: [item for item in lst if item not in li]
    )

    df_prob_dep = df_inter[["num_inter", "real_func"]].copy()
    df_inter = df_inter[
        ["Coord X", "Coord Y", "Month", "Day", "Hour", "Minute", "Duration", "Incident"]
    ]

    return df_inter, df_prob_dep, df_rank_incident


def clean_and_shift(row, cis_cols, li_new_cis):
    filtered_values = [
        val
        for val in row[cis_cols]
        if not (str(val).startswith("Z") or str(val).startswith("X"))
        and val not in li_new_cis
    ]
    new_values = filtered_values + [""] * (len(cis_cols) - len(filtered_values))
    row.update(pd.Series(new_values, index=cis_cols))
    return row


def union_iris(df_pdd, cis, li_new_cis):
    df_pdd = df_pdd[~df_pdd[cis].str.startswith("Z")]
    cis_cols = [col for col in df_pdd.columns if col.startswith("cis")]
    df_pdd = df_pdd.apply(
        clean_and_shift,
        args=(
            cis_cols,
            li_new_cis,
        ),
        axis=1,
    )

    area_name_unique = sorted(df_pdd[cis].unique())
    areas_geo = [
        MultiPolygon(list(df_pdd[df_pdd[cis] == nom].geometry))
        for nom in area_name_unique
    ]

    areas = gpd.GeoDataFrame(
        {"sector": area_name_unique, "geometry": areas_geo}, crs=df_pdd.crs
    )
    areas.geometry = areas.geometry.to_crs(2154)

    areas.geometry = areas.geometry.apply(lambda x: unary_union(x))

    return areas



def main() -> None:
    """
    Run the end-to-end preprocessing pipeline.

    This function reproduces the original top-level script behavior:
    - create expected project folders
    - load raw inputs from `./Data/`
    - run `preprocess(...)`
    - write outputs into `./Data_preprocessed/`
    """
    li_dir_name = [
        "Data_preprocessed",
        "Data_trained",
        "Data_sampled",
        "Data_environment",
        "SVG_model",
        "Plots",
        "Reward_weights",
    ]

    for dir_name in li_dir_name:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    os.chdir("./Data")

    csv_file = "inters.csv"
    df_inter = pd.read_csv(csv_file, sep=";")
    csv_file = "sorties.csv"
    df_sorties = pd.read_csv(csv_file, sep=";")
    filename = "pdd.geojson"
    df_pdd = gpd.read_file(filename)
    csv_file = "materiel.csv"
    df_materiel = pd.read_csv(csv_file, sep=";")

    df_real, df_prob_dep, df_rank_incident = preprocess(
        df_pdd, df_sorties, df_inter, df_materiel
    )

    svg_path = "../Data_preprocessed/"
    df_real.to_pickle(svg_path + "df_real.pkl")
    df_prob_dep.to_pickle(svg_path + "df_prob_dep.pkl")
    df_rank_incident.to_pickle(svg_path + "df_rank_incident.pkl")

    os.chdir("../")

    print(len(df_real), "Preprocess done")

    print(df_real.describe())


if __name__ == "__main__":
    main()
