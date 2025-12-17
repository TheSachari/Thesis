"""
collective_functions.py

Shared helper functions used by the emergency-response simulation environment and RL agents.

This module centralizes:
- environment loading (`load_environment_variables`)
- skill/role parsing and compatibility checks
- RL state construction (`gen_state`) and feasible-action extraction
- assignment step transitions (`step`) and reward/indicator helpers
- reinforcement / lending logistics for Z1 stations

The functions in this file frequently mutate dictionaries in-place (vehicles, planning,
duration trackers, etc.). When using them in training loops, copy inputs if you need a
pure/functional interface.
"""


import numpy as np
import re
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import json
import pickle
import torch

def get_skills_from_role(df_roles, role):
    """
    Return the skill requirement expression for a given role.

    Parameters
    ----------
    df_roles : pandas.DataFrame
        Roles/competences table. Expected to include at least:
        - a column containing role identifiers (e.g., "Role")
        - a column containing the competence expression (string) that can include
          plus/minus constraints and optional grade constraints.
    role : str
        Role identifier to look up in `df_roles`.

    Returns
    -------
    str
        The raw competence expression associated with `role`.

    Notes
    -----
    The returned expression is later parsed by `extract_skills` to produce:
    - required skills (plus / '+')
    - forbidden skills (minus / '-')
    - optional grade constraints.
    """

    return df_roles[(df_roles["Fonction"] == role)]["Competences"].reset_index(drop=True).values


def get_ff_compatible(df, station, ff_available, date, plus, minus):

    """
    Filter firefighters compatible with a role requirement at a given station and date.

    Parameters
    ----------
    df : pandas.DataFrame
        Skill validity table indexed by firefighter matricule. It must contain
        per-skill availability windows (start/end dates) for the firefighters.
    station : str
        Station (CIS) name where firefighters are selected.
    ff_available : list[int]
        List of firefighter identifiers currently available at the station and time slot.
    date : pandas.Timestamp or datetime-like
        Reference date/time used to check skill validity windows.
    plus : list[str]
        List of required skills that the firefighter must have.
    minus : list[str]
        List of forbidden skills that the firefighter must *not* have.

    Returns
    -------
    list[int]
        Subset of `ff_available` that satisfies all constraints.

    Notes
    -----
    This helper is used during assignment to check whether each candidate firefighter
    is eligible for the role being filled.
    """

    df_filtered = df.loc[ff_available, :]    
    mask_all_competences = pd.concat([(df_filtered[(comp, "Début")] <= date) & (df_filtered[(comp, "Fin")] >= date)
        for comp in plus], axis=1).all(axis=1)
    
    if len(df_filtered[mask_all_competences]) == 0:
        return []

    else:
        ff_plus = df_filtered[mask_all_competences].index.tolist()
        df_filtered_2 = df.loc[ff_plus, :]
        mask_nan_debut = df_filtered_2[[(comp, 'Début') for comp in minus]].isna()
        if len(df_filtered_2[mask_nan_debut.all(axis=1)]) == 0:
            return []
        else:            
            return df_filtered_2[mask_nan_debut.all(axis=1)].index.tolist()

def apply_logic(potential_actions, potential_skills, is_best):

    """
    Select an action among feasible actions using either a greedy ("best") or random policy.

    Parameters
    ----------
    potential_actions : list[int]
        Indices of feasible firefighter choices. Convention used in this project:
        - 0 can represent "do nothing / stop / no firefighter selected" in some contexts.
        - 79 is used as a sentinel for "no feasible firefighter found" in several pipelines.
    potential_skills : list[int]
        Skill-level indicators associated with each action (same order as `potential_actions`).
        Lower means "better" (e.g., exact match, less degradation).
    is_best : bool
        If True, select the action with the minimum skill level (ties broken arbitrarily).
        If False, sample a random action among `potential_actions`.

    Returns
    -------
    tuple[int, int]
        (action, skill_lvl) where:
        - action is the selected action index
        - skill_lvl is the associated skill level

    Notes
    -----
    This function does not apply the action; it only selects it. The action is later
    applied by `step` (environment mutation) or `lazy_step` (indicator-only update).
    """

    if is_best:
        action = potential_actions[potential_skills.index(min(potential_skills))]

    else:
        action = random.choice(potential_actions)
        
    skill_lvl = potential_skills[potential_actions.index(action)]

    return action, skill_lvl

def get_potential_actions(state, all_ff_waiting):

    """
    Compute feasible actions (firefighter indices) for the current role from the RL state matrix.

    Parameters
    ----------
    state : numpy.ndarray
        State matrix produced by `gen_state`. Expected layout:
        - row 0: global RL information / meta-features
        - row 1: one-hot encoding of the current role index
        - rows 2..: firefighter-specific features; the column corresponding to the current
          role encodes compatibility/skill indicators (>0 means feasible)
    all_ff_waiting : bool
        Whether the algorithm is in a "waiting for lent firefighters" mode. In that case,
        the only allowed action is typically action 0.

    Returns
    -------
    tuple[list[int], list[int]]
        (potential_actions, potential_skills)
        - potential_actions: feasible firefighter indices in the current consideration set
        - potential_skills: associated skill levels (same ordering)

    Notes
    -----
    - If no feasible firefighter exists, the function returns the sentinel action [79]
      and asserts that the last row of the state is all zeros (project-specific safety check).
    - When `all_ff_waiting` is True, the function asserts that `potential_actions == [0]`.
    """

    # 1st row: rl infos
    # 2nd row: idx role

    potential_actions = [79]
    skill_lvl = 0
    potential_skills = [0]
    cond_met = np.array([])
    # state = state.cpu().numpy()
    col_index = np.argmax(state[1, :] == 1) # current role
    column_values = state[2:, col_index] # ff available for a given role

    if not all_ff_waiting: # standard case
        selection = column_values[column_values > 0] # ff having the skill
        if selection.size > 0: # any ff ?        
            cond_met = np.where( (column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1)) )[0] # ff having any skill lvl > 0
            potential_skills = column_values[(column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1))].tolist()
    else: # all ff waiting
        selection = column_values # all ff to avoid the case of a ff losing his skills
        if selection.size > 0: # any ff ?
            cond_met = np.where( (state[2:, -2] == 1) )[0]                                               
            cond_met = np.array([cond_met[0]]) # first ff because all ff waiting follows an order
            
    if cond_met.size > 0:
        potential_actions = cond_met.tolist()
    else:
        potential_skills = [0]

    if potential_actions == [79]:
        assert np.all(state[-1] == 0), f"FF is on slot 79"

    if all_ff_waiting:
        assert potential_actions == [0], f"not action 0 and all_ff_waiting"

        
    return potential_actions, potential_skills

def get_v_availability(dic_vehicles, station):
    """
    Compute the vehicle availability ratio for a given station.

    Parameters
    ----------
    dic_vehicles : dict
        Station->vehicle pool dictionary created by `create_dic_vehicles`.
        Expected structure:
            dic_vehicles[station]["available"] : list
            dic_vehicles[station]["inter"] : list
            dic_vehicles[station]["standby"] : list
    station : str
        Station (CIS) name.

    Returns
    -------
    float
        Availability ratio in [0, 1], computed as:
            len(available) / (len(available) + len(inter) + len(standby))
    """

    v_available = len(dic_vehicles[station]["available"])

    v_all = sum([len(dic_vehicles[station][x]) for x in dic_vehicles[station]])

    return v_available/v_all

def get_ff_availability(planning, station, month, day, hour):
    """
    Compute the firefighter availability ratio for a station at a specific planning time slot.

    Parameters
    ----------
    planning : dict
        Nested planning dictionary loaded from `planning.pkl`:
            planning[station][month][day][hour] = {
                "planned": [...],
                "available": [...],
                "standby": [...],
            }
    station : str
        Station (CIS) name.
    month : int
        Month index (1..12).
    day : int
        Day of month (1..31).
    hour : int
        Hour of day (0..23).

    Returns
    -------
    float
        Availability ratio in [0, 1]. If the planning slot has no planned firefighters,
        returns 0 to avoid division by zero.
    """

    ff_available = len(planning[station][month][day][hour]['available'])
    ff_all = len(planning[station][month][day][hour]['planned'])
    if ff_all == 0:
        return 0
    else:
        return ff_available/ff_all

def get_neighborhood(pdd, station, num_d, n_following =5):

    """
    Return the list of following stations in a PDD (ordered candidate station list).

    Parameters
    ----------
    pdd : list[str]
        Ordered list of candidate stations for the current intervention.
    station : str
        Current station being considered.
    num_d : int
        Departure step number. This is used for heuristics and special handling of
        reinforcement steps (>= 99 in this codebase).
    n_following : int, default=5
        Maximum number of following stations to return.

    Returns
    -------
    list[str]
        Up to `n_following` stations following `station` in `pdd`.
    """

    idx = pdd.index(station)
    following = pdd[idx+1:idx+1+n_following]

    return following    

def get_neighborhood_availability(pdd, station, num_d, dic_vehicles, planning, month, day, hour, n_following):
    
    """
    Compute neighborhood availability features around a current station in the PDD list.

    Parameters
    ----------
    pdd : list[str]
        Ordered list of candidate stations for the intervention.
    station : str
        Current station.
    num_d : int
        Departure step number (used to determine which neighbors matter for specific steps).
    dic_vehicles : dict
        Station vehicle pools (`create_dic_vehicles`).
    planning : dict
        Planning dictionary with firefighter availability (`planning.pkl`).
    month, day, hour : int
        Time slot for firefighter availability lookup.
    n_following : int
        Number of following stations to consider.

    Returns
    -------
    numpy.ndarray
        1D feature vector summarizing vehicle and firefighter availability in the neighborhood.

    Notes
    -----
    This feature vector is appended into the RL state in `gen_state` and helps the policy
    account for nearby resources when deciding assignments.
    """

    if num_d < 79:
        info_avail = []
        neighborhood = get_neighborhood(pdd, station, num_d, n_following)
        for s in neighborhood:
            v_avail = get_v_availability(dic_vehicles, s)
            ff_avail = get_ff_availability(planning, s, month, day, hour)
            info_avail += [v_avail, ff_avail]
        for _ in range(n_following-len(neighborhood)):
            info_avail += [0, 0]
    else:
        info_avail = [0] * n_following * 2
    return info_avail

def load_environment_variables(constraint_factor_veh, constraint_factor_ff, dataset, start, end):

    """
    Load environment artifacts and slice the event stream to a given intervention interval.

    Parameters
    ----------
    constraint_factor_veh : int
        Vehicle constraint factor applied to Z1 stations (Toulouse Vion / Lougnon).
        A factor of 1 keeps all vehicles; factor k keeps roughly 1/k (random downsampling).
    constraint_factor_ff : int
        Firefighter constraint factor (random downsampling of the skill table).
    dataset : str
        Path (relative to `./Data_environment/`) to the pickled event stream DataFrame (df_pc).
    start : int
        First intervention id (num_inter) to include (departure event).
    end : int
        Last intervention id (num_inter) to include (RETURN event).

    Returns
    -------
    tuple
        (dic_vehicles, dic_functions, df_skills, dic_roles_skills, dic_roles, planning,
         dic_inter, dic_ff, dic_indic, dic_indic_old, Z_1, Z_4, dic_lent, dic_station_distance,
         df_pc, old_date, date_reference, skills_updated)

        Key structures:
        - dic_vehicles: station pools with keys ["available", "inter", "standby"]
        - dic_functions: vehicle_id -> list[str] of functions/types
        - df_skills: firefighter skill/validity table (possibly downsampled)
        - dic_roles_skills: mapping role->required skill vector
        - dic_roles: vehicle_type -> list[str] roles required
        - planning: nested availability dictionary planning[station][month][day][hour]
        - dic_inter: intervention log: num_inter -> station -> vehicle_id -> list[firefighters]
        - dic_ff: remaining duration per firefighter (minutes), updated over time
        - dic_indic: indicator counters used for reward/metrics
        - Z_1/Z_4: zone station lists
        - dic_lent: lent vehicle/firefighter structure for Z1 reinforcements
        - dic_station_distance: precomputed station distance ordering from Z1 to Z2/Z3
        - df_pc: sliced event stream between start and end (inclusive)
        - old_date/date_reference: reference timestamps for duration/skill updates
        - skills_updated: binary matrix of which skills are currently valid

    Side Effects
    ------------
    Changes working directory to `./Data_environment` during loading.

    Notes
    -----
    This is the canonical loader for both heuristic simulation and RL training/evaluation.
    """

    os.chdir('./Data_environment')

    df_stations = pd.read_pickle("df_stations.pkl")

    df_v = pd.read_pickle("df_v.pkl")
    dic_vehicles, dic_functions = create_dic_vehicles(df_v)
    dic_vehicles = purge_dic_v(dic_vehicles)

    print("constraint factor veh is ", constraint_factor_veh)
    
    list_of_mats = dic_vehicles["TOULOUSE - VION"]["available"]
    dic_vehicles["TOULOUSE - VION"]["available"] = constrain_veh(list_of_mats, constraint_factor_veh)
    
    list_of_mats = dic_vehicles["TOULOUSE - LOUGNON"]["available"]
    dic_vehicles["TOULOUSE - LOUGNON"]["available"] = constrain_veh(list_of_mats, constraint_factor_veh)

    df_skills = pd.read_pickle("df_skills.pkl")

    df_skills = df_skills.sample(len(df_skills)//constraint_factor_ff)
    print("constraint factor ff is ", constraint_factor_ff)
    
    df_roles = pd.read_pickle("df_roles.pkl")
    dic_roles_skills = generate_dic_roles_skills(df_roles, df_skills)

    df_vehicles_history = pd.read_pickle("df_vehicles_history.pkl")
    dic_roles = create_dic_roles(df_vehicles_history)
    
    with open("planning.pkl", "rb") as file:
        planning = pickle.load(file)

    df_pc = pd.read_pickle(dataset) #, sep = ';', parse_dates=["date"], converters={"PDD": ast.literal_eval, "departure": ast.literal_eval})

    dic_inter = {i:{} for i in range(1, int(len(df_pc)/2)+1)} # num_inter:station:mat_v:mat_ff
    
    dic_ff = {ff:0 for ff in df_skills.index}
    dic_indic = {'v_required': 0,
                    'v_sent': 0,
                    'v_sent_full':0,
                    'v_degraded':0,
                    'rupture_ff':0, #lack of ff
                    'function_not_found':0,
                    'v1_not_sent_from_s1':0,
                    'v3_not_sent_from_s3':0,
                    'v_not_found_in_last_station':0,
                    'ff_required':0,
                    'ff_sent':0,
                    'z1_VSAV_sent': 0,
                    'z1_FPT_sent': 0,
                    'z1_EPA_sent': 0,
                     'VSAV_needed':0,
                     'FPT_needed':0,
                     'EPA_needed':0,
                     'VSAV_disp':0,
                     'FPT_disp':0,
                     'EPA_disp':0,
                    'skill_lvl':0
                    } 
    dic_indic_old = dic_indic.copy()
    Z_1 = ['TOULOUSE - LOUGNON', 'TOULOUSE - VION']
    Z_2 = ['ST JORY', 'ROUFFIAC', 'RAMONVILLE - BUCHENS', 'COLOMIERS', 'MURET - MASSAT']
    Z_3 = ['AUTERIVE', 'ST LYS', 'GRENADE', 'FRONTON', 'VERFEIL', 'CARAMAN']
    Z_4 = [s for s in df_stations["Nom"] if s not in Z_1 + Z_2 + Z_3]
    print("Z_4", Z_4)
    dic_lent = {k:{} for k in Z_1} # station to, v_mat, ff_mat

    
    dic_station_distance = {ville_z1: trier_villes_par_distance(df_stations, ville_z1, Z_2 + Z_3) for ville_z1 in Z_1}

    idx_start = df_pc[(df_pc["num_inter"]==start) & (df_pc["departure"]!={0: 'RETURN'})].index[0]
    idx_end = df_pc[(df_pc["num_inter"]==end) & (df_pc["departure"]=={0: 'RETURN'})].index[0]
    df_pc = df_pc[idx_start:idx_end+1]

    print("df start-end", idx_start, idx_end)
    
    old_date = df_pc.iloc[0, 1]
    date_reference = df_pc.iloc[0, 1]
    skills_updated = update_skills(df_skills, date_reference)

    return dic_vehicles, dic_functions, df_skills, dic_roles_skills, dic_roles, planning, \
    dic_inter, dic_ff, dic_indic, dic_indic_old, Z_1, Z_4, dic_lent, dic_station_distance, df_pc, \
    old_date, date_reference, skills_updated


def gen_state(veh_depart, idx_role, ff_array, ff_existing, dic_roles, dic_roles_skills, dic_ff, df_skills, \
             coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size):

    """
    Build the state matrix used by the RL policy for firefighter assignment decisions.

    Parameters
    ----------
    veh_depart : list[str]
        Ordered list of requested vehicle functions/types for the current intervention.
    idx_role : int
        Index of the current role being assigned within the flattened role list of all vehicles.
    ff_array : numpy.ndarray
        Skill matrix for candidate firefighters, typically shape (N_ff, N_skills).
    ff_existing : list[int]
        Candidate firefighter identifiers corresponding to rows of `ff_array`.
    dic_roles : dict
        Mapping vehicle_type -> list[str] roles required (role names).
    dic_roles_skills : dict
        Mapping role_name -> skill requirement vector (aligned to df_skills columns).
    dic_ff : dict[int, int]
        Remaining duration (minutes) per firefighter; negative/zero values indicate unavailability.
    df_skills : pandas.DataFrame
        Skill table used to align skill vectors and compute compatibility.
    coord_x, coord_y : float
        Incident coordinates.
    month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos : float
        Periodic time embeddings used by the generative model / environment.
    info_avail : numpy.ndarray
        Neighborhood availability features computed by `get_neighborhood_availability`.
    max_duration : float
        Maximum intervention duration used to normalize the current duration feature.
    action_size : int
        Maximum number of firefighter "slots" encoded in the state (typically 80).

    Returns
    -------
    numpy.ndarray
        State matrix used by the policy and `get_potential_actions`.
        Convention in this project:
        - first rows encode global information and current role one-hot
        - remaining rows encode candidate firefighter features and compatibility indicators

    Notes
    -----
    The exact layout is project-specific; downstream code assumes:
    - row 1 is a one-hot indicator of the current role (used to select a column)
    - rows 2.. include per-firefighter role compatibility (>0 means feasible)
    """

    nb_roles = 37

    # ff skills
    state = np.hstack(([get_roles_for_ff(veh, ff_array, dic_roles, dic_roles_skills) for veh in veh_depart])).astype(float)

    state /= 8 # normalization, 8 skill lvls


    # filler row
    filler = np.zeros((action_size-state.shape[0], state.shape[1])) # max 74 de base + 6 ff lent
    state = np.vstack((state, filler))

    # filler col
    filler = np.zeros((state.shape[0], nb_roles - state.shape[1]))
    state = np.concatenate((state, filler), axis=1)

    # resp time
    resp_time = np.array([dic_ff[f] for f in df_skills.loc[ff_existing, :].index])
    resp_time_norm = np.where(resp_time < 0, 0.0, resp_time/max_duration) # normalization
    # resp_time_norm = np.where(resp_time < 0, 0, 1) # à voir avec mathieu pour l'AM
    mask_minus1 = (resp_time == -1) 
    mask_minus2 = (resp_time == -2)
    resp_time_all = np.stack([resp_time_norm, mask_minus1, mask_minus2], axis=1)

    zero_rows = np.zeros(((action_size-len(ff_existing)), resp_time_all.shape[1]))
    availability = np.vstack((resp_time_all, zero_rows))

    state = np.hstack((state, availability))

    # current role to fill
    current_role = [0]*state.shape[1]
    current_role[idx_role] = 1  
    # state = np.insert(state, 0, np.array(current_role), axis=0)
    state = np.vstack((current_role, state))

    # rl_infos + position + time

    rl_infos = np.array(info_avail + [coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos] + [0]*22)
    # print("rl_infos", rl_infos)
    state = np.vstack((rl_infos, state))

    # print("state shape final", state.shape, flush=True)

    return state

def constrain_veh(list_of_mats, factor=3, seed=42):
    """
    Randomly downsample a list of vehicles by a given factor.

    Parameters
    ----------
    list_of_mats : list
        List of vehicle IDs/material IDs.
    factor : int, default=3
        Downsampling factor. For factor k, keeps roughly 1/k of the list.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    list
        Downsampled list of vehicles.
    """

    random.seed(seed)
    random.shuffle(list_of_mats)
    size_subset = len(list_of_mats) // factor
    return list_of_mats[:size_subset]

def get_start_hour(df, num_inter):

    """
    Return the (month, day, hour) of an intervention start event from the event stream.

    Parameters
    ----------
    df : pandas.DataFrame
        Event stream DataFrame (df_pc) containing at least columns:
        - "num_inter" : intervention id
        - "departure" : dict, where a departure event is not {0: 'RETURN'}
        - "Month", "Day", "Hour" : time components for the event
    num_inter : int
        Intervention id to locate.

    Returns
    -------
    tuple[int, int, int]
        (month, day, hour) for the first departure event of `num_inter`.
    """

    idx = df[df["num_inter"] == num_inter].index[0]
    date = df["date"].loc[idx]

    return date.month, date.day, date.hour

def compute_reward(dic_indic, dic_indic_old, num_d, dic_tarif):
    """
    Compute the reward for a decision step from indicator deltas and a tariff/weight dictionary.

    Parameters
    ----------
    dic_indic : dict
        Current indicator counters.
    dic_indic_old : dict
        Previous indicator counters (baseline for delta computation).
    num_d : int
        Departure step number (used to apply step-specific weights in some setups).
    dic_tarif : dict
        Weight dictionary mapping indicator names (and possibly step keys) to scalar weights.

    Returns
    -------
    float
        Scalar reward computed from weighted indicator deltas.

    Notes
    -----
    This helper is used in RL training/evaluation loops to transform operational indicators
    (vehicles sent, degraded departures, firefighter shortages, etc.) into a scalar objective.
    """

    reward = 0

    if num_d < 79:

        dic_delta = {key:(dic_indic[key] - dic_indic_old[key]) for key in dic_indic if key not in ['VSAV_disp', 'FPT_disp', 'EPA_disp']}

        for m in dic_delta:

            reward += dic_delta[m] * dic_tarif[m]

        
        if dic_indic['VSAV_disp'] < 2:
            reward += dic_tarif['VSAV_disp']
    
        if dic_indic['FPT_disp'] < 2:
            reward += dic_tarif['FPT_disp']
    
        if dic_indic['EPA_disp'] < 1:
            reward += dic_tarif['EPA_disp']


    return reward

def lazy_step(action, num_d, num_role, mandatory, degraded, new_dic_indic, skill_lvl):

    """
    Update indicator counters for an action without mutating environment structures.

    Parameters
    ----------
    action : int
        Selected action index.
    num_d : int
        Departure step number.
    num_role : int
        Role index within the current vehicle team.
    mandatory : int
        Number of mandatory roles for the current vehicle type.
    degraded : bool
        Whether the current vehicle is already degraded.
    new_dic_indic : dict
        Indicator dictionary to update (copied by caller).
    skill_lvl : int
        Skill mismatch/gradation level for the chosen firefighter.

    Returns
    -------
    dict
        Updated indicator dictionary.

    Notes
    -----
    This is useful for computing rewards or simulating outcomes without performing
    the full in-place assignment (`step`), e.g., during action evaluation.
    """

    if action < 79:
        
        new_dic_indic['skill_lvl'] += skill_lvl * 8  # was normalized in state
        
    else: # aucun pompier n'a les compétences requises

        
        if (num_role > mandatory) and (num_d == 1): # Si le rôle est facultatif et que c'est 
        # le 1er véhicule 

            degraded = True

        else: # Si le rôle n'est pas facultatif ou que ce n'est pas le 1er véhicule    

            new_dic_indic['rupture_ff'] += 1

    return new_dic_indic


def step(action, idx_role, ff_existing, all_ff_waiting, current_station, Z_1, dic_lent, \
    v_mat, dic_ff, VSAV_lent, FPT_lent, EPA_lent, planning, month, day, hour, num_inter, new_required_departure, num_d, \
    list_v, num_role, mandatory, degraded, team_max, all_roles_found, \
    vehicle_found, dic_vehicles, dic_indic, skill_lvl, station_lvl):

    """
    Apply a firefighter assignment action for the current role and update environment state.

    Parameters
    ----------
    action : int
        Selected action index from the feasible action set.
    idx_role : int
        Current flattened role index (will be advanced as roles are filled).
    ff_existing : list[int]
        Candidate firefighter identifiers.
    all_ff_waiting : bool
        Whether the algorithm is waiting for lent firefighters to return/become available.
    current_station : str
        Station currently being processed in the PDD loop.
    Z_1 : list[str]
        List of Z1 stations (special handling for reinforcements/lending).
    dic_lent : dict
        Lent structure mapping Z1 station -> {vehicle_id: [firefighters]}.
    v_mat : str
        Vehicle/material identifier being assigned.
    dic_ff : dict[int, int]
        Remaining durations per firefighter (mutated when firefighters are assigned).
    VSAV_lent, FPT_lent, EPA_lent : bool
        Flags tracking whether reinforcements have been lent (per vehicle type).
    planning : dict
        Planning dictionary (mutated: available/standby lists).
    month, day, hour : int
        Time slot for availability updates.
    num_inter : int
        Current intervention id.
    new_required_departure : dict
        Departure dict for remaining vehicles to dispatch (mutated when failures occur).
    num_d : int
        Current departure step.
    list_v : list[str]
        List of acceptable vehicle function types for this step (OR-list).
    num_role : int
        Current role number within the vehicle.
    mandatory : int
        Number of mandatory roles required for the vehicle to be considered non-degraded.
    degraded : bool
        Current degraded status for the vehicle.
    team_max : int
        Max team size / role count for the vehicle.
    all_roles_found : bool
        Whether all roles were successfully assigned for the vehicle.
    vehicle_found : bool
        Whether a physical vehicle was successfully selected.
    dic_vehicles : dict
        Station vehicle pools (mutated to move vehicles between available/standby/inter).
    dic_indic : dict
        Indicator counters (mutated).
    skill_lvl : int
        Skill-level indicator for the chosen firefighter.
    station_lvl : int
        Index of the current station in the PDD search.

    Returns
    -------
    tuple
        (dic_indic, dic_lent, all_roles_found, vehicle_found, planning, dic_vehicles,
         dic_ff, idx_role, degraded)

    Side Effects
    ------------
    Mutates multiple dictionaries in-place (planning, dic_vehicles, dic_ff, dic_lent, dic_indic).

    Notes
    -----
    This is the core transition function for the assignment process and is used by both
    heuristic simulation and RL-driven decision-making.
    """

    if action < 79:
        idx_role += 1

        ff_mat = ff_existing[action]
        dic_indic['skill_lvl'] += skill_lvl * 8  # was normalized in state
        
        if all_ff_waiting and (current_station in Z_1): # pompiers à rapatrier
            dic_lent[current_station][v_mat].remove(ff_mat)
            dic_ff[ff_mat] = -2 # was already in standby -1

        else:
            dic_ff[ff_mat] = -1

            if not (VSAV_lent or FPT_lent or EPA_lent or (current_station in Z_1)):
                planning[current_station][month][day][hour]['available'].remove(ff_mat)
          
        planning[current_station][month][day][hour]['standby'].append(ff_mat)

    else: # aucun pompier n'a les compétences requises
        new_required_departure[num_d] = list_v # Le véhicule requis est ajouté au nouveau train
        # si le rôle est obligatoire ou que ce n'est pas le 1er véhicule
        
        if (num_role > mandatory) and (num_d == 1): # Si le rôle est facultatif et que c'est 
        # le 1er véhicule 
            if not degraded: 
                degraded = True
                # print(v_mat, "degraded")
            idx_role += 1

        else: # Si le rôle n'est pas facultatif ou que ce n'est pas le 1er véhicule    

            all_roles_found, vehicle_found, planning, dic_vehicles, \
            dic_ff = cancel_departure(all_roles_found, vehicle_found, planning, current_station, \
                                      month, day, hour, dic_vehicles, dic_ff, v_mat)  
            idx_role += team_max - (num_role -1) # le num_role est itéré une fois de plus au-delà du max
            dic_indic['rupture_ff'] += 1
            # dic_indic['ff_skill_lvl'][v_mat] = []

    return dic_indic, dic_lent, all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff, idx_role, degraded

def get_mandatory_max(v):
    """
    Return the number of mandatory roles and the max team size for a given vehicle type.

    Parameters
    ----------
    v : str
        Vehicle function/type label (e.g., "VSAV", "FPT", "EPA", ...).

    Returns
    -------
    tuple[int, int]
        (mandatory, team_max)
        - mandatory: number of mandatory roles that must be filled to avoid degradation
        - team_max: maximum number of roles in the vehicle team
    """

    mandatory = team_max = 2
    
    if "VSAV" in v:
        mandatory, team_max = 2, 3
    elif "FPT" in v :
        mandatory, team_max = 4, 6
    elif "EP" in v:
        mandatory, team_max = 2, 3
    elif "VSR" in v:
        mandatory, team_max = 2, 3
    elif "PCC" in v:
        mandatory, team_max = 1, 1
    # elif any(w in v for w in ["VID","VBAL","VTUTP", "VGD"]):
    #     mandatory, team_max = 2, 2
    elif v == "CCF":
        mandatory, team_max = 3, 4
    # elif "CCFL" in v:
    #     mandatory, team_max = 2, 3
       

    return mandatory, team_max

def create_dic_roles(df_vehicles_history):

    """
    Build a dictionary mapping vehicle function/type to the ordered list of roles.

    Parameters
    ----------
    df_vehicles_history : pandas.DataFrame
        Vehicle history table containing at least the vehicle type and role information.

    Returns
    -------
    dict[str, list[str]]
        Mapping vehicle_type -> list of role names in the order they should be assigned.
    """

    df_vehicles_history["Fonction"] = df_vehicles_history["Fonction"].fillna("")
    dic_replace = {"XCOMPL":"COMPL"}
    df_vehicles_history["Fonction"] = df_vehicles_history["Fonction"].replace(dic_replace)
    
    dic_roles = {}

    for _, row in df_vehicles_history.iterrows():

        tm = row["Type Matériel"]
        t = row["Type"]
        f = row["Fonction"]
        ofo = row["Ordre Fonction Occupee"]
        fo = row["Fonction Occupee"]
        
        if (tm != ""):
            if tm not in dic_roles:
                dic_roles[tm] = {ofo:fo}
            elif ofo not in dic_roles[tm]:
                dic_roles[tm][ofo] = fo
        if (t != ""):
            if t not in dic_roles:
                dic_roles[t] = {ofo:fo}
            elif ofo not in dic_roles[t]:
                dic_roles[t][ofo] = fo
        if (f != ""):
            if f not in dic_roles:
                dic_roles[f] = {ofo:fo}
            elif ofo not in dic_roles[f]:
                dic_roles[f][ofo] = fo

    dic_roles = {
    fonction: {num_role: role
                  for num_role, role in valeurs.items() if num_role <= 6}
    for fonction, valeurs in dic_roles.items()
}

    dic_roles = {k: v for k, v in dic_roles.items() if not (isinstance(k, float) and np.isnan(k))}

    for veh, nb_ro in dic_roles.items():
        mandatory, team_max = get_mandatory_max(veh)
        dic_roles[veh] = {k: v for k, v in dic_roles[veh].items() if k <= team_max}

    return dic_roles


def get_potential_veh(Z_1, dic_vehicles, dic_functions, v_type):

    """
    Compute how many vehicles of a given type are available in Z1 and choose a target station.

    Parameters
    ----------
    Z_1 : list[str]
        Z1 stations list.
    dic_vehicles : dict
        Station vehicle pools.
    dic_functions : dict
        Vehicle_id -> list[str] of functions.
    v_type : str
        Target vehicle type/function to count (e.g., "VSAV", "FPT", "EPA").

    Returns
    -------
    tuple[int, str]
        (v_disp, v_to_station)
        - v_disp: number of available vehicles of that type across Z1 stations
        - v_to_station: station in Z1 that should receive reinforcement (heuristic choice)
    """

    # 2 VSAV en Z1
    # 2 FPT + 1 EPC en Z1
    v_disp_tl = len([item for item in dic_vehicles['TOULOUSE - LOUGNON']["available"] \
                     if any(v_type in func for func in dic_functions[item])]) # or func.startswith(v_type)
    v_disp_tv = len([item for item in dic_vehicles['TOULOUSE - VION']["available"] \
                     if any(v_type in func for func in dic_functions[item])]) #  or func.startswith(v_type)
    v_to_station = Z_1[np.argmin([v_disp_tl, v_disp_tv])]
    return (v_disp_tl+v_disp_tv), v_to_station

def update_dict(dic, k):
    """
    Increment a counter in a dictionary for a given key.

    Parameters
    ----------
    dic : dict
        Dictionary of counters.
    k : hashable
        Key to increment.

    Returns
    -------
    dict
        The same dictionary, updated in-place (counter incremented by 1).
    """

    if k in dic:
        dic[k] += 1
    else:
        dic[k] = 1
    return dic

def get_role_from_skills(required_skills, ff_array):

    """
    Find the first role index compatible with a given firefighter's skills for a vehicle.

    Parameters
    ----------
    skills : numpy.ndarray
        Binary skill vector for a firefighter (aligned to df_skills columns).
    role_skills : list[numpy.ndarray]
        List of required-skill vectors for each role of the vehicle.
    mandatory : int
        Number of mandatory roles (used to determine degradation thresholds).

    Returns
    -------
    tuple[int, int]
        (role_idx, skill_lvl)
        - role_idx: index of the first compatible role (or a sentinel if none)
        - skill_lvl: compatibility/gradation level for that assignment
    """

    matches_minus_one = (required_skills == -1)[:, np.newaxis, :]  # Reshape pour broadcast
    matches_one = (required_skills == 1)[:, np.newaxis, :]
    matches_zero_or_any = (required_skills == 0)[:, np.newaxis, :]
    
    conditions_met = ((matches_minus_one & (ff_array == 0)[np.newaxis, :, :]) | 
                      (matches_one & (ff_array == 1)[np.newaxis, :, :]) | 
                      matches_zero_or_any)
    
    conditions_met = np.all(conditions_met, axis=2)
    
    first_valid_index = np.argmin(conditions_met, axis=0) +1
    
    first_valid_index[~np.any(conditions_met, axis=0)] = 0
    
    return first_valid_index

def get_roles_for_ff(vehicle, ff_array, dic_roles, dic_roles_skills):

    """
    Compute compatible roles (and skill levels) for each firefighter for each vehicle in the departure list.

    Parameters
    ----------
    veh_depart : list[str]
        Ordered list of vehicle types for the intervention.
    ff_array : numpy.ndarray
        Firefighter skill matrix aligned to df_skills columns.
    dic_roles : dict
        vehicle_type -> list[str] roles.
    dic_roles_skills : dict
        role_name -> skill requirement vectors.
    mandatory : int
        Number of mandatory roles per vehicle (or vector derived per vehicle).

    Returns
    -------
    tuple
        A structure encoding, for each firefighter and each role, whether it is feasible and at what skill level.
    """

    required_roles = dic_roles[vehicle]

    required_roles = [required_roles[k] for k in sorted(required_roles.keys())]
    
    # required_roles = [role if role in dic_roles_skills else 'EQ_ENG_SAP' for role in required_roles]

    return np.column_stack([get_role_from_skills(dic_roles_skills[role], ff_array).reshape(-1, 1) for role in required_roles])

def distance_euclidienne(x1, y1, x2, y2):
    """
    Compute Euclidean distance between two points.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Point coordinates.

    Returns
    -------
    float
        Euclidean distance.
    """

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def trier_villes_par_distance(df, ville_z1, villes_z2):

    """
    Sort stations by distance from a reference station.

    Parameters
    ----------
    df : pandas.DataFrame
        Stations table with columns ["Nom", "Coordonnée X", "Coordonnée Y"].
    ville_z1 : str
        Reference station name.
    villes_z2 : list[str]
        Candidate stations to sort.

    Returns
    -------
    dict[str, int]
        Mapping station -> rounded distance in kilometers, sorted ascending.
    """

    x1, y1 = df.loc[df['Nom'] == ville_z1, ['Coordonnée X', 'Coordonnée Y']].values[0]
    

    distances = []
    for ville_z2 in villes_z2:
        x2, y2 = df.loc[df['Nom'] == ville_z2, ['Coordonnée X', 'Coordonnée Y']].values[0]
        dist = distance_euclidienne(x1, y1, x2, y2)
        dist /= 1000
        distances.append((ville_z2, int(dist)))
    
    # Trier les villes de z2 par distance croissante
    distances.sort(key=lambda x: x[1])  # Trie par la distance (le deuxième élément)
    
    # Retourner les villes triées et leurs distances
    return {ville:dist for ville, dist in distances}

def get_skill_array(plus, minus, df_skills, zeros=134):
    """
    Encode a role requirement into a {-1, 0, 1} vector aligned with df_skills skill columns.

    Parameters
    ----------
    df_skills : pandas.DataFrame
        Skill table defining the canonical skill column ordering.
    plus : list[str]
        Required skills (+).
    minus : list[str]
        Forbidden skills (-).

    Returns
    -------
    numpy.ndarray
        Vector of length N_skills where:
        - 1 indicates a required skill
        - -1 indicates a forbidden skill
        - 0 indicates no constraint
    """

    idx_plus = df_skills.columns.get_level_values(0).unique().get_indexer(plus)
    idx_minus = df_skills.columns.get_level_values(0).unique().get_indexer(minus)
    array = np.zeros(zeros, dtype=int)
    array[idx_plus] = 1
    array[idx_minus] = -1
    return array

def extract_skills(s):

    """
    Parse a competence expression string into plus/minus constraints and optional grade constraints.

    Parameters
    ----------
    skills : str
        Raw competence expression (from `df_roles`) containing skill tokens and operators.

    Returns
    -------
    tuple
        (plus, minus, grade_constraints)
        - plus: list of required skills
        - minus: list of forbidden skills
        - grade_constraints: project-specific representation of grade/level constraints (if any)

    Notes
    -----
    The exact grammar is defined implicitly by the project data (roles_competences.csv).
    """

    plus = [re.findall(r"[\w]+", s)[0]]
    plus += re.findall(r"\+\s*[\w]+", s)
    plus = [w.replace("+ ", "") for w in plus]
    minus = re.findall(r"-\s*[\w]+", s)
    minus = [w.replace("- ", "") for w in minus]
    grade = re.findall(r"s*GRADE\([A-Z]+\)", s)
    
    if grade:
        grade = grade[0][4:-1]
        sup = re.findall(r">", s)        
        if sup:            
            sup = sup[0]
        else:
            inf = re.findall(r"<", s)[0]    
    else:
        sup = ""

    return  pd.Series([plus, minus, grade, sup])

def generate_dic_roles_skills(df_roles, df_skills):
    
    """
    Create a mapping from role name to required skill vector aligned with df_skills columns.

    Parameters
    ----------
    df_roles : pandas.DataFrame
        Roles/competences table containing role names and competence expressions.
    df_skills : pandas.DataFrame
        Skill table defining canonical skill ordering and firefighter indices.

    Returns
    -------
    dict[str, numpy.ndarray]
        role_name -> constraint vector of length N_skills (values in {-1,0,1}).
    """

    df_roles["Competences"] = df_roles["Competences"].fillna("")
    
    df_roles[["Plus", "Minus", "Grade", "Sup"]] = df_roles["Competences"].apply(extract_skills)
    df_roles["Required_roles"] = df_roles.apply(lambda row: get_skill_array(row['Plus'], row['Minus'], df_skills), axis=1)
    dic_roles_skills = {}
    for fonction, group in df_roles.groupby('Fonction'):
        sorted_group = group.sort_values(by='Ordre')
        dic_roles_skills[fonction] = np.vstack(sorted_group['Required_roles'].tolist())
    return dic_roles_skills

def create_dic_vehicles(df_v):

    """
    Create station vehicle pools and a vehicle->functions mapping from the vehicles table.

    Parameters
    ----------
    df_v : pandas.DataFrame
        Vehicles inventory table, typically loaded from `df_v.pkl`, containing:
        - station name ("Nom du Centre")
        - vehicle/material identifier ("IU Materiel")
        - function/type information ("Fonction materiel" / "Type materiel")

    Returns
    -------
    tuple[dict, dict]
        (dic_vehicles, dic_functions)
        - dic_vehicles[station] = {"available": [...], "inter": [...], "standby": [...]}
        - dic_functions[vehicle_id] = list[str] functions/types associated with the vehicle
    """

    dic_vehicles = {}
    dic_functions = {}
    for station, group in df_v.groupby('Nom du Centre'):
        dic_vehicles[station] = {"available" : group['IU Materiel'].tolist(), 
                                 "standby" : [], 
                                 "inter" : [], 
                                 "VSAV_sent":[], 
                                 "FPT_sent":[], 
                                 "EPA_sent":[]}

    dic_functions = dict(zip(df_v['IU Materiel'], df_v['Fonction materiel']))

    return dic_vehicles, dic_functions

def purge_dic_v(dic):
    """
    Remove duplicated vehicle IDs across station pools.

    Parameters
    ----------
    dic : dict
        Station vehicle pools.

    Returns
    -------
    dict
        Cleaned station vehicle pools with duplicates removed.

    Notes
    -----
    This enforces the invariant that a physical vehicle belongs to one station pool only.
    """

    v = []
    new_dic = {}
    
    for s, d in dic.items():
        new_dic[s] = {}
        for st, li_v in d.items():
            new_dic[s][st] = []
            for e in li_v:
                if e not in v:
                    v.append(e)
                    new_dic[s][st].append(e)

    return new_dic

def reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, dic_ff, dic_log, planning, v_from_station, \
    v_to_station, v_sent, v_returning, v_lent, v_to_return, month, day, hour, dic_start_time, v_type):
    
    """
    Handle the arrival of a lent reinforcement vehicle and update station availability.

    Parameters
    ----------
    num_inter : int
        Current intervention id (event index).
    dic_vehicles : dict
        Station vehicle pools (mutated).
    dic_back : dict
        Bookkeeping structure tracking return trips for reinforcements (mutated).
    dic_lent : dict
        Lent structure mapping destination station -> {vehicle_id: [firefighters]} (mutated).
    dic_ff : dict[int, int]
        Remaining duration per firefighter (mutated).
    dic_log : dict
        Log structure tracking reinforcement events (mutated).
    planning : dict
        Planning availability structure (mutated).
    v_from_station : str
        Origin station of the reinforcement.
    v_to_station : str
        Destination station.
    v_sent : bool
        Whether the reinforcement is currently in transit.
    v_returning : bool
        Whether the reinforcement is returning.
    v_lent : bool
        Whether the vehicle is considered lent at destination.
    v_to_return : bool
        Whether the vehicle should return to origin after use.
    month, day, hour : int
        Time slot for planning updates.
    dic_start_time : dict
        Map vehicle_id -> (month, day, hour) start time.
    v_type : str
        Vehicle type ("VSAV", "FPT", "EPA") for logging and routing.

    Returns
    -------
    tuple
        Updated (dic_vehicles, v_sent, v_lent, v_returning, dic_ff, planning, dic_back,
        dic_lent, dic_log, v_to_return)

    Side Effects
    ------------
    Mutates multiple dictionaries in-place to reflect the reinforcement arrival.
    """

    veh_mat = dic_vehicles[v_from_station][v_type + "_sent"][0]
    dic_vehicles[v_to_station]["available"].append(veh_mat)
    dic_vehicles[v_from_station][v_type + "_sent"] = [] 
    v_sent -= 1 

    start_month, start_day, start_hour = dic_start_time[veh_mat]
    
    if v_returning: 
        v_lent -= 1
        v_returning = False
        for f in dic_back[veh_mat]:
            dic_ff[f] = 0
            if (f in planning[v_to_station][month][day][hour]["planned"]) and \
            (f not in planning[v_to_station][month][day][hour]["available"]):
                planning[v_to_station][month][day][hour]["available"].append(f)

            if (f not in planning[v_to_station][start_month][start_day][start_hour]["available"]):
                planning[v_to_station][start_month][start_day][start_hour]["available"].append(f)

        
        # print(num_inter, v_type, veh_mat, dic_back[veh_mat], "sent back from", v_from_station, "to", v_to_station, "has arrived")
        del dic_back[veh_mat]  
        del dic_lent[v_from_station][veh_mat]   
        del dic_log[veh_mat]

    else: 
        v_lent += 1
        for f in dic_lent[v_to_station][veh_mat]:
            dic_ff[f] = 0

        # print(num_inter, v_type, veh_mat, dic_lent[v_to_station][veh_mat], "sent from", v_from_station, "to", v_to_station, "has arrived")

    v_to_return = False

    return dic_vehicles, v_sent, v_lent, v_returning, dic_ff, planning, dic_back, dic_lent, dic_log, v_to_return

def returning(df_pc, dic_inter, num_inter, vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, month, day, hour):

    """
    Handle a RETURN event for an intervention and restore resources.

    Parameters
    ----------
    df_pc : pandas.DataFrame
        Event stream.
    dic_inter : dict
        Intervention log mapping num_inter -> station -> vehicle_id -> list[firefighters].
    num_inter : int
        Current intervention id.
    vehicle_out : int
        Counter of vehicles currently out (used for tracking).
    dic_vehicles : dict
        Station vehicle pools (mutated to move vehicles back to available).
    dic_ff : dict[int, int]
        Remaining durations per firefighter (mutated).
    current_ff_inter : list[int]
        List of firefighters currently engaged (mutated).
    planning : dict
        Planning structure (mutated to restore availability).
    month, day, hour : int
        Time slot of the return event.

    Returns
    -------
    tuple
        (vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, dic_inter)

    Notes
    -----
    This is called when `departure == {0: "RETURN"}` in the event stream.
    """

    start_month, start_day, start_hour = get_start_hour(df_pc, num_inter)

    # print("returning before", [f for f in dic_ff if dic_ff[f]==-1])
    for station, mats in dic_inter[num_inter].items(): # num_inter:station:v_mat:ff_mat          
        for veh_mat, ff_mats in mats.items():
            vehicle_out -= 1
            
            dic_vehicles[station]["inter"].remove(veh_mat)
            dic_vehicles[station]["available"].append(veh_mat)

            for f in ff_mats:
                dic_ff[f] = 0
                current_ff_inter.remove(f)
                
                if (f in planning[station][month][day][hour]["planned"]) and \
                (f not in planning[station][month][day][hour]["available"]):
                    planning[station][month][day][hour]["available"].append(f)
                    # print("f", f, "has been added in available", month, day, hour)

                if (f not in planning[station][start_month][start_day][start_hour]["available"]):
                    planning[station][start_month][start_day][start_hour]["available"].append(f)
                    # print("f", f, "has been added to start time")
         
            # print(num_inter, "vehicle in", station, veh_mat, ff_mats, vehicle_out)
                 
            dic_inter[num_inter][station][veh_mat] = []

    # print("returning after", [f for f in dic_ff if dic_ff[f]==-1])
    return vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, dic_inter

def veh_management(v_disp, v_needed, v_to_return, v_lent, v_to_station, \
    new_required_departure, dic_station_distance, num_inter, dic_lent, \
    dic_vehicles, dic_functions, dic_ff, threshold, v_type, new_num_d):

    """
    Determine whether a reinforcement vehicle is needed, and update departure requirements accordingly.

    Parameters
    ----------
    v_disp : int
        Number of currently available vehicles of the target type in Z1.
    v_needed : bool
        Whether reinforcement is currently needed (flag).
    v_to_return : bool
        Whether a vehicle should be returned (flag).
    v_lent : bool
        Whether a vehicle is lent at destination (flag).
    v_to_station : str
        Destination station for the potential reinforcement.
    new_required_departure : dict
        Dict of required departures to be updated when reinforcements are inserted/removed.
    dic_station_distance : dict
        Precomputed distance ordering from Z1 to other zones.
    num_inter : int
        Current intervention id.
    dic_lent : dict
        Lent structure (mutated).
    dic_vehicles : dict
        Station vehicle pools (mutated).
    dic_functions : dict
        Vehicle_id -> functions.
    dic_ff : dict[int, int]
        Firefighter durations (mutated in some cases).
    threshold : int
        Minimum number of available vehicles considered sufficient.
    v_type : str
        Vehicle type ("VSAV", "FPT", "EPA").
    new_num_d : int
        Special departure step number used for reinforcements (e.g., 99/100/101).

    Returns
    -------
    tuple
        (stations_iter, v_needed, v_to_return, new_required_departure, v_to_station, dic_ff, v_mat_to_return)

    Notes
    -----
    This function encodes the reinforcement policy for Z1 stations.
    """

    stations_v = iter([])
    v_mat = 0
    # print(v_disp, v_disp, threshold, threshold, )
    if (v_disp < threshold):
        v_needed = True
        v_to_return = False
        new_required_departure[new_num_d] = [v_type]
        stations_v = iter(dic_station_distance[v_to_station])                                                    
        # print(num_inter, v_type, "needed for station", v_to_station)
        
    elif (v_disp > threshold) and v_lent: # libération du véhicule 
        v_needed = False                 
        flag = 0
        
        # Mise en attente des pompiers envoyés: 
        v_to_station = [s for s, vff in dic_lent.items() if vff and any(v_type in dic_functions[v] for v in vff)][0]
        # print("v_to_station", v_to_station)
        stations_v = iter([v_to_station])
        # print(num_inter, v_type, "not needed anymore for station", v_to_station)      
        for veh_mat, ff_mats in dic_lent[v_to_station].items(): 
            if veh_mat in dic_vehicles[v_to_station]["available"] and v_type in dic_functions[veh_mat]:
                # print(num_inter, v_type, veh_mat, "to return is available")
                v_to_return = True
                v_mat = veh_mat
                new_required_departure[new_num_d] = [v_type]
                for ff in ff_mats:
                    if dic_ff[ff] == 0:
                        dic_ff[ff] = -1
                        # print("ff", ff, "to return is available")
                if all(dic_ff[ff] == -1 for ff in ff_mats):
                    # print(num_inter, "all ff waiting for", v_type, veh_mat, ff_mats)
                    flag = 1
                # else:
                #     print("not all ff waiting")

            if flag:
                break

    return stations_v, v_needed, v_to_return, new_required_departure, v_to_station, dic_ff, v_mat
            
def cancel_departure(all_roles_found, vehicle_found, planning, current_station, \
    month, day, hour, dic_vehicles, dic_ff, v_mat):
    
    """
    Cancel a departure when mandatory roles cannot be filled and roll back allocations.

    Parameters
    ----------
    all_roles_found : bool
        Whether all roles were assigned.
    vehicle_found : bool
        Whether the vehicle was selected.
    dic_lent : dict
        Lent structure (mutated).
    current_station : str
        Station where the departure is being processed.
    month, day, hour : int
        Time slot.
    dic_vehicles : dict
        Station vehicle pools (mutated).
    dic_ff : dict[int, int]
        Firefighter durations (mutated).
    v_mat : str
        Vehicle/material id.

    Returns
    -------
    tuple
        Updated (dic_lent, dic_vehicles, dic_ff, all_roles_found, vehicle_found)
    """

    all_roles_found = True
    vehicle_found = True
    # Les pompiers sortent du standby
    ff_mats = planning[current_station][month][day][hour]['standby'].copy()
    planning[current_station][month][day][hour]['standby'] = []
    # Le véhicule en standby est à nouveau disponible
    dic_vehicles[current_station]["standby"].remove(v_mat)
    dic_vehicles[current_station]["available"].append(v_mat) 
    # les pompiers sont à nouveau disponibles
    # print(v_mat, "cancel departure")
    for f in ff_mats:
        # if dic_ff[f] == -2:
        #     dic_ff[f] = -1
        #     # print(f, "is in standby again")
        # else:  
        dic_ff[f] = 0
        # print(f, "is available again") 
        if (f in planning[current_station][month][day][hour]["planned"]) and \
        (f not in planning[current_station][month][day][hour]["available"]):
            planning[current_station][month][day][hour]["available"].append(f)
        # print("f", f, "has been added in available", month, day, hour)
        

    # print("cancel departure after:", [f for f in dic_ff if dic_ff[f]<0])

    return all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff

def reinforcement_returning(num_inter, v_to_station, v_from_station, dic_log, v_mat, dic_vehicles, \
    dic_station_distance, date, df_pc, idx, dic_back, ff_to_send, v_needed, \
    v_sent, all_ff_waiting, v_waiting, v_returning, v_type):
    
    """
    Send a lent vehicle back to its origin station and schedule its arrival.

    Parameters
    ----------
    num_inter : int
        Current intervention id.
    v_to_station : str
        Current destination station (where vehicle is lent).
    v_from_station : str
        Origin station.
    dic_log : dict
        Log structure updated with return metadata.
    v_mat : str
        Vehicle/material id returning.
    dic_vehicles : dict
        Station vehicle pools (mutated).
    dic_station_distance : dict
        Distance ordering used to schedule arrival.
    date : pandas.Timestamp
        Current event timestamp.
    df_pc : pandas.DataFrame
        Event stream (used to infer future arrival index).
    idx : int
        Current event row index.
    dic_back : dict
        Structure tracking return trips (mutated).
    ff_to_send : list[int]
        Firefighters on the returning vehicle.
    v_needed : bool
        Reinforcement-needed flag.
    v_sent : bool
        Vehicle-sent flag.
    all_ff_waiting : bool
        Waiting flag.
    v_waiting : bool
        Vehicle waiting flag.
    v_returning : bool
        Returning flag.
    v_type : str
        Vehicle type.

    Returns
    -------
    tuple
        Updated set of routing flags and structures, including next arrival intervention number.
    """

    v_from_station = v_to_station
    v_to_station = dic_log[v_mat]
    dic_vehicles[v_from_station][v_type+"_sent"].append(v_mat)
    arrival_time = date + timedelta(minutes = dic_station_distance[v_from_station][v_to_station] + 20)
    arrival_num = df_pc.loc[(df_pc.index >= idx) & (df_pc["date"] >= arrival_time), "num_inter"].iloc[0]
    # print(num_inter, v_type, v_mat, ff_to_send, "sent back from", v_from_station, "will arrive at", arrival_num, "to", v_to_station)
    dic_back[v_mat] = ff_to_send
    dic_log[v_mat] = v_from_station
    # print("reinforcement_returning", dic_log)
    v_needed = False
    v_sent += 1
    all_ff_waiting = False
    v_waiting = False
    v_to_return = False
    v_returning = True

    return v_from_station, v_to_station, arrival_num, dic_back, dic_log, v_needed, v_sent, all_ff_waiting, v_waiting, v_to_return, v_returning

def reinforcement_sending(num_inter, current_station, v_from_station, v_mat, dic_vehicles, \
    dic_station_distance, v_to_station, date, df_pc, idx, dic_lent, \
    ff_to_send, dic_log, v_needed, v_sent, \
    required_departure, new_required_departure, num_d, v_type):
    
    """
    Send a vehicle as reinforcement to a Z1 station and schedule its arrival.

    Parameters
    ----------
    num_inter : int
        Current intervention id.
    current_station : str
        Station currently being processed (often Z1 destination).
    v_from_station : str
        Origin station for the reinforcement.
    v_mat : str
        Vehicle/material id being sent.
    dic_vehicles : dict
        Vehicle pools (mutated).
    dic_station_distance : dict
        Distance ordering used to estimate routing/arrival.
    v_to_station : str
        Destination station.
    date : pandas.Timestamp
        Current timestamp.
    df_pc : pandas.DataFrame
        Event stream.
    idx : int
        Current event row index.
    dic_lent : dict
        Lent structure (mutated).
    ff_to_send : list[int]
        Firefighters assigned to the reinforcement.
    dic_log : dict
        Log structure (mutated).
    v_needed : bool
        Flag indicating a reinforcement is needed.
    v_sent : bool
        Flag indicating a reinforcement is currently in transit.
    required_departure : dict
        Current departure requirements.
    new_required_departure : dict
        Mutable dict for additional requirements.
    num_d : int
        Departure step number.
    v_type : str
        Vehicle type.

    Returns
    -------
    tuple
        Updated (v_from_station, dic_vehicles, arrival_num, dic_lent, dic_log,
        new_required_departure, v_needed, v_sent)
    """

    v_from_station = current_station
    dic_vehicles[v_from_station][v_type+"_sent"].append(v_mat)
    arrival_time = date + timedelta(minutes = dic_station_distance[v_to_station][v_from_station] + 20)
    arrival_num = df_pc.loc[(df_pc.index >= idx) & (df_pc["date"] >= arrival_time), "num_inter"].iloc[0]
    # print(num_inter, v_type, v_mat, ff_to_send, "sent from", v_from_station, "will arrive at", arrival_num, "to", v_to_station)
    dic_lent[v_to_station][v_mat] = ff_to_send.copy() # mise à disposition des pompiers
    dic_log[v_mat] = v_from_station
    # print("reinforcement_sending", dic_log)

    if num_d in new_required_departure:
        del new_required_departure[num_d]

    v_needed = False
    v_sent += 1

    return v_from_station, dic_vehicles, arrival_num, dic_lent, dic_log, new_required_departure, v_needed, v_sent 

def v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, current_station, dic_vehicles, v_type, v_mat_to_return):

    """
    Select the appropriate physical vehicle when a reinforcement vehicle is 'to return'.

    Parameters
    ----------
    dic_log : dict
        Reinforcement log structure.
    li_mat_veh : list[str]
        Candidate vehicle/material IDs at the station that match the requested type.
    v_waiting : bool
        Whether the algorithm is currently waiting on the returning vehicle.
    vehicle_to_find : str
        Vehicle function/type being searched (e.g., "VSAV").
    current_station : str
        Station being processed.
    dic_vehicles : dict
        Vehicle pools.
    v_type : str
        Vehicle type label.
    v_mat_to_return : str
        Vehicle/material ID that is expected to return (if known).

    Returns
    -------
    tuple[str, bool]
        (v_mat, v_waiting) selected material id and updated waiting flag.
    """

    v_to_return = [v for v in li_mat_veh if (v in dic_log)]
    if v_to_return and v_mat_to_return != 0:
        # v_mat = v_to_return[0]
        v_mat = v_mat_to_return
        v_waiting = True
        # print("managing", vehicle_to_find, "to return", v_mat, "found in station", current_station) 
    else:         
        v_mat = li_mat_veh[0]
        v_waiting = False
        # print("managing", v_type, "to return NOT found in station", current_station, "now looking for", vehicle_to_find, v_mat)
    return v_mat, v_waiting

def adding_lent_ff(VSAV_lent, FPT_lent, EPA_lent, current_station, Z_1, dic_lent, ff_mats, dic_ff):

    """
    Augment the current station firefighter list with lent firefighters (if applicable).

    Parameters
    ----------
    VSAV_lent, FPT_lent, EPA_lent : bool
        Flags indicating whether each reinforcement type is currently lent.
    current_station : str
        Station name.
    Z_1 : list[str]
        Z1 stations list.
    dic_lent : dict
        Lent structure mapping station -> {vehicle_id: [firefighters]}.
    ff_mats : list[int]
        Current firefighter list at the station (mutated / extended).
    dic_ff : dict[int, int]
        Firefighter durations; used to filter out unavailable firefighters.

    Returns
    -------
    list[int]
        Updated firefighter list including lent resources.
    """

    if (VSAV_lent or FPT_lent or EPA_lent) and (current_station in Z_1): # ajout des renforts à la station de Z1

        ff_lent = [f for v in dic_lent[current_station] if v in dic_lent[current_station] \
                   for f in dic_lent[current_station][v]]
        ff_mats += ff_lent
        ff_mats = list(set(ff_mats))

    return [f for f in ff_mats if f in dic_ff].copy() # Pour éviter les pompiers manquants


def are_all_ff_waiting(ff_existing, current_station, dic_lent, dic_ff, v_mat):
        
    """
    Check whether all lent firefighters associated with a vehicle are waiting/standby.

    Parameters
    ----------
    ff_existing : list[int]
        Candidate firefighter list currently considered.
    current_station : str
        Station name.
    dic_lent : dict
        Lent structure.
    dic_ff : dict[int, int]
        Firefighter remaining durations.
    v_mat : str
        Vehicle/material id.

    Returns
    -------
    bool
        True if all lent firefighters for `v_mat` are waiting (project-specific definition).
    """

    ff_waiting = [f for f in ff_existing if (dic_ff[f] == -1)] # already put in standby -1
    return all(f in ff_waiting for f in dic_lent[current_station][v_mat])   

    # else:
    #     lent_ff = [ff for v_mat in dic_lent.values() for ff_lent in v_mat.values() for ff in ff_lent]
    #     ff_not_lent = [num for num in ff_existing if num not in lent_ff]
    #     ff_existing = [f for f in ff_not_lent if dic_ff[f] > -1].copy()





# def get_required_roles(dic_roles, required_departure):
#     return [val for dico in [dict(sorted(dic_roles[vehicle_to_find].items())) \
#                                  for vehicle_to_find in [v[0] for k, v in required_departure.items()]] for val in dico.values()]

def gen_ff_array(df_skills, skills_updated, ff_existing):
    """
    Extract the skill matrix for a given subset of firefighters.

    Parameters
    ----------
    df_skills : pandas.DataFrame
        Skill table indexed by firefighter matricule.
    skills_updated : numpy.ndarray or pandas.DataFrame
        Binary availability/skill matrix at the current reference date (output of `update_skills`).
    ff_existing : list[int]
        Firefighter identifiers to select.

    Returns
    -------
    numpy.ndarray
        Skill matrix for the selected firefighters aligned to df_skills columns.
    """

    return skills_updated[[df_skills.index.get_loc(matricule) for matricule in ff_existing]]

    
def update_duration(date, old_date, current_ff_inter, dic_ff):

    """
    Update remaining durations for currently engaged firefighters based on elapsed time.

    Parameters
    ----------
    date : pandas.Timestamp
        Current event timestamp.
    old_date : pandas.Timestamp
        Previous event timestamp.
    current_ff_inter : list[int]
        Firefighters currently engaged in interventions.
    dic_ff : dict[int, int]
        Remaining duration per firefighter in minutes (mutated).

    Returns
    -------
    dict[int, int]
        Updated `dic_ff` with decremented durations based on time delta.

    Notes
    -----
    This function is called at each event step to progress ongoing interventions.
    """

    elapsed_time = (date - old_date).total_seconds() / 60    
    for f in current_ff_inter:
        dic_ff[f] -= int(elapsed_time)

    return dic_ff

def update_skills(df_skills, date_reference):
    """
    Compute a binary "skills_updated" matrix indicating which skills are valid at a reference date.

    Parameters
    ----------
    df_skills : pandas.DataFrame
        Skills table with validity windows (start/end dates) per firefighter and skill.
    date_reference : pandas.Timestamp
        Reference date used to determine validity.

    Returns
    -------
    numpy.ndarray
        Binary matrix indicating current skill validity for each firefighter and skill,
        aligned to the ordering used by `df_skills`.
    """

    condition_list = []
    for col in df_skills.columns.get_level_values(0).unique():
        deb_col = (col, 'Début')
        fin_col = (col, 'Fin')    
        condition = (df_skills[deb_col] <= date_reference) & (df_skills[fin_col] >= date_reference)
        condition_list.append(condition.rename(col))
        
    conditions = pd.concat(condition_list, axis=1)
    return np.where(conditions.fillna(False), 1, 0)

def update_dep(required_departure):
    """
    Reindex a departure dictionary so that step numbers are consecutive.

    Parameters
    ----------
    required_departure : dict[int, list]
        Departure dict mapping step -> list of acceptable vehicle functions.
        Steps < 99 correspond to standard departure steps in this codebase.

    Returns
    -------
    dict[int, list]
        New departure dict with steps renumbered consecutively starting at 1 (for steps < 99),
        while preserving any special reinforcement steps (>= 99).
    """

    new_d = {}
    new_k = 1
    for k, v in sorted(required_departure.items()):
        if k < 99:
            new_d[new_k] = v
            new_k += 1
        else:
            new_d[k] = v
    
    return new_d
