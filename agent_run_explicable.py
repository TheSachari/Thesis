"""
agent_run_explicable.py

Run an "explainable" RL agent (e.g., FQF) inside the emergency-response simulation loop.

This script is primarily intended for:
- Evaluating a trained agent on an event stream (`df_pc_real.pkl` or `df_pc_fake.pkl`)
- Optionally training online (when `--train` is provided)
- Tracking and saving simulation metrics and reward evolution

High-level workflow
-------------------
1) Parse CLI arguments (hyper-parameters JSON, dataset, constraints, model paths, reward weights)
2) Load agent hyper-parameters and instantiate the requested agent (FQF currently supported here)
3) Load model weights and switch the policy network to eval mode (this file currently runs eval-only by default)
4) Load reward weights (tariffs) and environment variables/event stream
5) Iterate over events and:
   - handle RETURN events (restore resources)
   - handle departure events by selecting vehicles, assigning firefighters and applying the agent policy
   - compute reward and optionally train via agent.step(...)
6) Periodically log metrics and save model checkpoints

Import safety
-------------
The executable portion is wrapped in `main()` and guarded by `if __name__ == "__main__":`,
so importing this module does not start a run.

Notes
-----
- This code mutates environment dictionaries in-place (vehicles, planning, etc.).
- The original file contained a syntax error in the tuple unpacking of `row` (extra commas).
  This refactor fixes that unpacking so the script can run.
"""

import pandas as pd
import numpy as np
from agent import *
from collective_functions import *

from datetime import datetime, timedelta
import torch
import json
import argparse
import re
import ast
import os
import pickle
from IPython.display import clear_output



def main() -> None:
    """Run the agent loop (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Agent parameters")
    # parser.add_argument("--hyper_params", type=str, help="Agent hyper parameters")
    parser.add_argument("--hyper_params", type=str, help="Agent hyper parameters")
    parser.add_argument("--dataset", type=str, default="df_pc_fake.pkl", help="Name of dataset")
    parser.add_argument("--eps_start", type=float, help="epsilon", default=1.0)
    parser.add_argument("--train", action='store_true', help="Train mode")
    parser.add_argument("--load", action='store_true', help="Load weights")
    parser.add_argument("--agent_model", type=str, help="Model name")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--start", type=int, default=1, help="start from num_inter")
    parser.add_argument("--end", type=int, default=159264, help="end after num_inter")
    parser.add_argument("--reward_weights", type=str, help="JSON file with reward weights")
    parser.add_argument("--save_metrics_as", type=str, default="dic_indic_agent", help="save metrics as")
    parser.add_argument("--constraint_factor_veh", type=int, default=1, help="size of available vehicles in Z1. factor 1 is 100%, factor 3 is 33%")
    parser.add_argument("--constraint_factor_ff", type=int, default=1, help="size of available firefighters. factor 1 is 100%, factor 3 is 33%")

    args = parser.parse_args()

    os.chdir('./Data')

    hyper_params = json.load(open(args.hyper_params, "r"))
    eps = args.eps_start
    compute = False

    device = torch.device(hyper_params["device"])
    torch.autograd.set_detect_anomaly(True)
    hyper_params["max_train_steps"] = (args.end-args.start) * 5 # (approx. 5 actions by intervention)
    print("max_train_steps", hyper_params["max_train_steps"])
    if args.agent_model == "fqf":
        agent = FQF_Agent(**hyper_params)



    eps = 0
    os.chdir('../SVG_model')
    agent.qnetwork_local.load_state_dict(torch.load(args.model_name, weights_only=True))
    agent.qnetwork_local.eval()

    print("Eval ONLY - weights loaded", flush=True)  

    os.chdir('../Reward_weights')

    dic_tarif = json.load(open(args.reward_weights))
    # print("Reward weights", dic_tarif)

    os.chdir('../')

    ### LOAD ENVIRONMENT VARIABLES ###

    dic_vehicles, dic_functions, df_skills, dic_roles_skills, dic_roles, planning, dic_inter, \
    dic_ff, dic_indic, dic_indic_old, Z_1, Z_4, dic_lent, dic_station_distance, df_pc, old_date, date_reference, \
    skills_updated = load_environment_variables(args.constraint_factor_veh, args.constraint_factor_ff, \
                                                args.dataset, args.start, args.end)

    vehicle_out, num_d, score, action_num = 0, 42, 0, 0
    all_ff_waiting, v_waiting, following_depart = (False,) * 3

    vehicle_evo, reward_evo, current_ff_inter = [], [], []
    dic_log, dic_back, dic_start_time, dic_veh_typ = {}, {}, {}, {}

    VSAV_to_station, VSAV_from_station = "", ""
    FPT_to_station, FPT_from_station = "", ""
    EPA_to_station, EPA_from_station = "", ""
    VSAV_sent, VSAV_lent, VSAV_needed, VSAV_to_return, VSAV_returning = (False,) * 5
    FPT_sent, FPT_lent, FPT_needed, FPT_to_return, FPT_returning = (False,) * 5
    EPA_sent, EPA_lent, EPA_needed, EPA_to_return, EPA_returning = (False,) * 5
    VSAV_disp, FPT_disp, EPA_disp = (0,)*3

    eps_update = (args.end-args.start) // 23 # approx. 23 iterations to reach 5% of original eps
    d = 1
    print("eps_start", eps, "eps_update", eps_update, flush=True)

    max_duration = df_pc["Duration"].max()
    action_size = hyper_params["action_size"] # idx role + rl infos
    dic_indic_100 = dic_indic.copy()

    os.chdir('../SVG_model')


    # for idx, inter in tqdm(df_pc.iloc[:-20].itertuples(index=True, name=None), total=len(df_pc.iloc[:-20])):
    for row in df_pc.itertuples(index=True, name=None):

        idx, num_inter, date, pdd, required_departure, zone, duration, month, day, hour, minute, \
        coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos  = row

        dic_ff = update_duration(date, old_date, current_ff_inter, dic_ff)

        if date > date_reference:
            date_reference = date
            skills_updated = update_skills(df_skills, date_reference)

            counts = np.count_nonzero(skills_updated, axis=0)

            print(counts)
            break

        if (VSAV_sent) and (num_inter == VSAV_arrival_num) :  # ARRIVEE DES RENFORTS VSAV
            dic_vehicles, VSAV_sent, VSAV_lent, VSAV_returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            VSAV_to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, VSAV_from_station, VSAV_to_station, VSAV_sent, \
                                                    VSAV_returning, VSAV_lent, VSAV_to_return, month, day, hour, dic_start_time, "VSAV")

        if (FPT_sent) and (num_inter == FPT_arrival_num) :  # ARRIVEE DES RENFORTS FPT
            dic_vehicles, FPT_sent, FPT_lent, FPT_returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            FPT_to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, FPT_from_station, FPT_to_station, FPT_sent, \
                                                    FPT_returning, FPT_lent, FPT_to_return, month, day, hour, dic_start_time, "FPT")

        if (EPA_sent) and (num_inter == EPA_arrival_num) :  # ARRIVEE DES RENFORTS EPA
            dic_vehicles, EPA_sent, EPA_lent, EPA_returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            EPA_to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, EPA_from_station, EPA_to_station, EPA_sent, \
                                                    EPA_returning, EPA_lent, EPA_to_return, month, day, hour, dic_start_time, "EPA")

        if (required_departure == {0:"RETURN"}):  # RETOUR D'INTERVENTION    
            vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, \
            dic_inter = returning(df_pc, dic_inter, num_inter, vehicle_out, dic_vehicles, \
                                  dic_ff, current_ff_inter, planning, month, day, hour)

        else: # INTERVENTION      
            veh_depart = [v[0] for k, v in sorted(required_departure.items())]

            dic_indic['v_required'] += len(required_departure)

            if len(veh_depart) > 5:
                new_veh_depart = veh_depart[5:].copy()
                veh_depart = veh_depart[:5]  
                required_departure = {i + 1: [v] for i, v in enumerate(veh_depart)}
                following_depart = True
                # print("following_depart")

            new_required_departure = {}
            stations = iter(pdd)            
            inter_done = False
            VSAV_to_return, FPT_to_return, EPA_to_return = (False,) * 3
            station_lvl = 0
            idx_role = 0
            v_mat_to_return = 0

            vehicle_evo.append([num_inter, vehicle_out])

            while not inter_done: # Tant que l'intervention n'est pas finie   

                current_station = next(stations, False) # On va dans la plus proche caserne
                if (num_d < 99) and (current_station not in dic_inter[num_inter]):
                    # Il s'agit d'une intervention et non d'un renfort
                    dic_inter[num_inter][current_station] = {}

                station_lvl += 1

                if not current_station: # S'il n'y a plus de plus proche caserne, l'intervention est terminée
                    inter_done = True
                    departure_done = True         
                    dic_indic['v_not_found_in_last_station'] += len(required_departure)
                    # print("v_not_found_in_last_station")

                else: # Sinon on cherche les véhicules requis
                    departure_done = False
                    required_vehicles = iter(sorted(required_departure.items()))

                # print(num_inter, required_departure)

                while not departure_done: # Tant que tous les véhicules n'ont pas été envoyés

                    num_d, list_v = next(required_vehicles, (0, [])) # On cherche les véhicules requis dans le train initial
                    # print("start", "num_d", num_d, "list_v", list_v)
                    if station_lvl == 2 and num_d == 1:
                        dic_indic['v1_not_sent_from_s1'] += 1
                        # print("v1_not_sent_from_1st_station", station_lvl)
                    if station_lvl > 3 and num_d <= 3 and current_station in Z_4:
                        dic_indic['v3_not_sent_from_s3'] += 1

                    if list_v:
                        mandatory, team_max = get_mandatory_max(list_v[0])

                    if not list_v: # S'il n'y a plus de véhicules requis dans le train initial    
                        departure_done = True # On a envoyés tous les véhicules possibles depuis cette caserne 
                        # print("no more v to send from this station")
                        if new_required_departure: # S'il reste des véhicules à faire partir dans le nouveau train  
                            required_departure = new_required_departure
                            required_departure = update_dep(required_departure) # to test
                            veh_depart = [v[0] for k, v in sorted(required_departure.items())]
                            new_required_departure = {}
                            idx_role = 0
                            # print("new_required_departure", veh_depart)

                        else: # S'il ne reste pas de véhicules à faire partir
                            inter_done = True # L'intervention est terminée
                            if following_depart:
                                inter_done = False
                                stations = iter(pdd)
                                station_lvl = 0
                                required_departure = {i + 1: [v] for i, v in enumerate(new_veh_depart)}
                                required_departure = update_dep(required_departure)
                                veh_depart = new_veh_depart

                                if len(veh_depart) > 5:
                                    new_veh_depart = veh_depart[5:].copy()
                                    veh_depart = veh_depart[:5]  
                                    required_departure = {i + 1: [v] for i, v in enumerate(veh_depart)}
                                    following_depart = True
                                    # print("following_depart", veh_depart)

                                new_required_departure = {}
                                idx_role, num_d = 0, 1
                                following_depart = False

                    else: # S'il y a des véhicules requis dans le train initial   

                        if (num_d == 99):
                            current_station = next(stations_VSAV, False)
                        if (num_d == 100):
                            current_station = next(stations_FPT, False)
                        if (num_d == 101):
                            current_station = next(stations_EPA, False)
                        # print(num_inter, "current_station", current_station, "num_d", num_d)
                        # print("to return", VSAV_to_return, FPT_to_return, EPA_to_return)

                        vehicles_to_find = iter(list_v)   
                        vehicle_found = False
                        vehicle_lvl = 1

                        if not current_station:
                            vehicle_found = True
                            # v_waiting = False
                            # print('no station')
                            all_ff_waiting = False

                        while not vehicle_found: # Tant qu'on a pas trouvé le véhicule requis

                            vehicle_to_find = next(vehicles_to_find, False) # On cherche la prochaine fonction à faire partir  

                            if not vehicle_to_find: # S'il n'y a plus de fonction à faire partir

                                idx_role += team_max 
                                vehicle_found = True
                                new_required_departure[num_d] = list_v # Le véhicule requis est ajouté au nouveau train 
                                dic_indic['function_not_found'] += 1
                                # print("function not found")

                            else: # S'il y a une fonction à faire partir depuis cette caserne
                                # On cherche les véhicules disponibles dans la caserne actuelle correspondant à la fonction requise  

                                # print(num_inter, "v_mat before li_mat_veh", v_mat_to_return)

                                v_mats = dic_vehicles[current_station]["available"].copy()                             
                                li_mat_veh = [v_m for v_m in v_mats if vehicle_to_find in dic_functions[v_m]]
                                # li_mat_veh = [v_mat for v_mat in v_mats if any(func.startswith(vehicle_to_find) for func in dic_functions[v_mat])]

                                if li_mat_veh: # Si des véhicules ont la fonction requise   

                                    if VSAV_to_return and (current_station == VSAV_to_station) and (num_d == 99):
                                        v_mat, v_waiting = v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles, "VSAV", v_mat_to_return)
                                        # print("VSAV_to_return", v_mat, v_waiting)

                                    elif FPT_to_return and (current_station == FPT_to_station) and (num_d == 100):
                                        v_mat, v_waiting = v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles, "FPT", v_mat_to_return)
                                        # print("FPT_to_return", v_mat, v_waiting)

                                    elif EPA_to_return and (current_station == EPA_to_station) and (num_d == 101):
                                        v_mat, v_waiting = v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles, "EPA", v_mat_to_return) 
                                        # print("EPA_to_return", v_mat, v_waiting)

                                    else: 
                                        v_mat = li_mat_veh[0] 

                                    if num_d < 99:
                                        veh_depart[num_d-1] = vehicle_to_find

                                    # print(num_inter, "veh_depart", veh_depart, "num_d", num_d, vehicle_to_find, v_mat, "found in", current_station)

                                    mandatory, team_max = get_mandatory_max(vehicle_to_find)

                                    # On met le véhicule en standby
                                    dic_vehicles[current_station]["available"].remove(v_mat)
                                    dic_vehicles[current_station]["standby"].append(v_mat) 
                                    vehicle_found = True
                                    # print(num_inter, "vehicle_found", dic_functions[v_mat],  v_mat, "in",  current_station, "num_d", num_d)

                                    # On cherche les rôles à pourvoir
                                    roles = dic_roles[vehicle_to_find]
                                    # print(roles)
                                    role_number = iter(range(1, len(roles) + 1))
                                    all_roles_found = False
                                    degraded = False

                                    while not all_roles_found: # Tant que tous les rôles ne sont pas pourvus

                                        num_role = next(role_number, (0))  

                                        if num_role > team_max: # Pour limiter les rôles
                                            num_role = 0

                                        if not num_role: # S'il n'y a plus de rôles à pourvoir   
                                            # print("no more role to fill", v_waiting)
                                            all_roles_found = True
                                            ff_to_send = planning[current_station][month][day][hour]['standby'].copy()
                                            planning[current_station][month][day][hour]['standby'] = []    
                                            dic_vehicles[current_station]["standby"].remove(v_mat)

                                            if VSAV_needed and (num_d == 99): # DEPART VSAV EN RENFORT
                                                dic_start_time[v_mat] = (month, day, hour)
                                                VSAV_from_station, dic_vehicles, VSAV_arrival_num, dic_lent, \
                                                dic_log, new_required_departure, VSAV_needed, \
                                                VSAV_sent  = reinforcement_sending(num_inter, current_station, VSAV_from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   VSAV_to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, VSAV_needed, \
                                                                                   VSAV_sent, required_departure, \
                                                                                   new_required_departure, num_d, "VSAV")
                                                dic_indic['z1_VSAV_sent'] += 1

                                            elif FPT_needed and (num_d == 100): # DEPART FPT EN RENFORT
                                                dic_start_time[v_mat] = (month, day, hour)
                                                FPT_from_station, dic_vehicles, FPT_arrival_num, dic_lent, \
                                                dic_log, new_required_departure, FPT_needed, \
                                                FPT_sent  = reinforcement_sending(num_inter, current_station, FPT_from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   FPT_to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, FPT_needed, \
                                                                                   FPT_sent, required_departure, \
                                                                                   new_required_departure, num_d, "FPT")
                                                dic_indic['z1_FPT_sent'] += 1

                                            elif EPA_needed and (num_d == 101): # DEPART EPA EN RENFORT
                                                dic_start_time[v_mat] = (month, day, hour)
                                                EPA_from_station, dic_vehicles, EPA_arrival_num, dic_lent, \
                                                dic_log, new_required_departure, EPA_needed, \
                                                EPA_sent  = reinforcement_sending(num_inter, current_station, EPA_from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   EPA_to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, EPA_needed, \
                                                                                   EPA_sent, required_departure, \
                                                                                   new_required_departure, num_d, "EPA")
                                                dic_indic['z1_EPA_sent'] += 1
                                                # print(num_inter, "EPA", EPA_sent)

                                            elif v_waiting: # RENFORT A RETOURNER

                                                # print("v_waiting, all_ff_waiting", all_ff_waiting)

                                                if all_ff_waiting: # RETOUR DES RENFORTS

                                                    if VSAV_to_return and (num_d == 99):  # RETOUR DU VSAV
                                                        VSAV_from_station, VSAV_to_station, VSAV_arrival_num, dic_back, dic_log, \
                                                        VSAV_needed, VSAV_sent, all_ff_waiting, v_waiting, VSAV_to_return, \
                                                        VSAV_returning = reinforcement_returning(num_inter, VSAV_to_station, VSAV_from_station, \
                                                                                                 dic_log, \
                                                                                                 v_mat, dic_vehicles, dic_station_distance, date,\
                                                                                                 df_pc, idx, dic_back, ff_to_send, VSAV_needed, \
                                                                                                 VSAV_sent, all_ff_waiting, v_waiting, \
                                                                                                 VSAV_returning, "VSAV")

                                                    elif FPT_to_return and (num_d == 100): # RETOUR DU FPT
                                                        FPT_from_station, FPT_to_station, FPT_arrival_num, dic_back, dic_log, \
                                                        FPT_needed, FPT_sent, all_ff_waiting, v_waiting, FPT_to_return, \
                                                        FPT_returning = reinforcement_returning(num_inter, FPT_to_station, FPT_from_station, \
                                                                                                dic_log, \
                                                                                                v_mat, dic_vehicles, dic_station_distance, date,\
                                                                                                 df_pc, idx, dic_back, ff_to_send, FPT_needed, \
                                                                                                 FPT_sent, all_ff_waiting, v_waiting, \
                                                                                                 FPT_returning, "FPT")

                                                    elif EPA_to_return and (num_d == 101):  # RETOUR DE L'EPA
                                                        EPA_from_station, EPA_to_station, EPA_arrival_num, dic_back, dic_log, \
                                                        EPA_needed, EPA_sent, all_ff_waiting, v_waiting, EPA_to_return, \
                                                        EPA_returning = reinforcement_returning(num_inter, EPA_to_station, EPA_from_station, \
                                                                                                dic_log, \
                                                                                                v_mat, dic_vehicles, dic_station_distance, date,\
                                                                                                 df_pc, idx, dic_back, ff_to_send, EPA_needed, \
                                                                                                 EPA_sent, all_ff_waiting, v_waiting, \
                                                                                                 EPA_returning, "EPA")

                                                else: # retour impossible, pompiers indisponibles

                                                    # print("vehicle", vehicle_to_find, v_mat, 'available again in', current_station)
                                                    v_waiting = False
                                                    dic_vehicles[current_station]["available"].append(v_mat)
                                                    planning[current_station][month][day][hour]['available'] += ff_to_send

                                            else: # départ de véhicule en inter

                                                dic_inter[num_inter][current_station][v_mat] = ff_to_send
                                                dic_vehicles[current_station]["inter"].append(v_mat)

                                                current_ff_inter += ff_to_send
                                                for f in ff_to_send:
                                                    dic_ff[f] = duration

                                                vehicle_out += 1

                                                # print(num_inter, "vehicle out", current_station, v_mat, ff_to_send, vehicle_out, "|", VSAV_lent,"/", VSAV_disp, "|", FPT_lent,"/",FPT_disp, "|", EPA_lent,"/",EPA_disp)

                                                dic_indic['v_sent'] += 1
                                                if degraded: 
                                                    dic_indic['v_degraded'] += 1 
                                                else:
                                                    dic_indic['v_sent_full'] += 1 
                                                dic_indic['ff_sent'] += len(ff_to_send)

                                                dic_veh_typ = update_dict(dic_veh_typ, vehicle_to_find) # metrique

                                                if (current_station in Z_1): # GESTION DES RENFORTS

                                                    if not VSAV_sent: # s'il n'y a pas de renfort en route
                                                        VSAV_disp, VSAV_to_station = get_potential_veh(Z_1, dic_vehicles, dic_functions, "VSAV") 

                                                        stations_VSAV, VSAV_needed, VSAV_to_return, new_required_departure, VSAV_to_station, \
                                                        dic_ff, v_mat_to_return = veh_management(VSAV_disp, VSAV_needed, VSAV_to_return, VSAV_lent, VSAV_to_station, \
                                                                                new_required_departure, dic_station_distance, num_inter, dic_lent, \
                                                                                dic_vehicles, dic_functions, dic_ff, 2, "VSAV", 99) 

                                                        dic_indic['VSAV_needed'] += int(VSAV_needed)
                                                        dic_indic['VSAV_disp'] = int(VSAV_disp)
                                                        # print(num_inter, "v_mat", v_mat_to_return)

                                                    elif not FPT_sent:
                                                        FPT_disp, FPT_to_station = get_potential_veh(Z_1, dic_vehicles, dic_functions, "FPT")

                                                        stations_FPT, FPT_needed, FPT_to_return, new_required_departure, FPT_to_station, \
                                                        dic_ff, v_mat_to_return = veh_management(FPT_disp, FPT_needed, FPT_to_return, FPT_lent, FPT_to_station, \
                                                                                new_required_departure, dic_station_distance, num_inter, dic_lent, \
                                                                                dic_vehicles, dic_functions, dic_ff, 2, "FPT", 100) 

                                                        dic_indic['FPT_needed'] += int(FPT_needed)
                                                        dic_indic['FPT_disp'] = int(FPT_disp)
                                                        # print(num_inter, "v_mat", v_mat_to_return)

                                                    elif not EPA_sent:
                                                        EPA_disp, EPA_to_station = get_potential_veh(Z_1, dic_vehicles, dic_functions, "EPA")

                                                        stations_EPA, EPA_needed, EPA_to_return, new_required_departure, EPA_to_station, \
                                                        dic_ff, v_mat_to_return = veh_management(EPA_disp, EPA_needed, EPA_to_return, EPA_lent, EPA_to_station, \
                                                                                new_required_departure, dic_station_distance, num_inter, dic_lent, \
                                                                                dic_vehicles, dic_functions, dic_ff, 1, "EPA", 101)   

                                                        dic_indic['EPA_needed'] += int(EPA_needed)
                                                        dic_indic['EPA_disp'] = int(EPA_disp)
                                                        # print(num_inter, "v_mat", v_mat_to_return)


                                        else: # S'il y a un rôle à pourvoir

                                            info_avail = get_neighborhood_availability(pdd, current_station, num_d, dic_vehicles, \
                                                                                       planning, month, day, hour, 5) 

                                            ff_mats = planning[current_station][month][day][hour]["planned"].copy()

                                            ff_existing = adding_lent_ff(VSAV_lent, FPT_lent, EPA_lent, \
                                                                         current_station, Z_1, dic_lent, ff_mats, dic_ff)  

                                            if v_waiting:

                                                # print("role to fill and v_waiting")

                                                v_mat = v_mat_to_return

                                                if VSAV_to_return and (current_station==VSAV_to_station) and "VSAV" in dic_functions[v_mat]:    
                                                    all_ff_waiting = are_all_ff_waiting(ff_existing, current_station, \
                                                                                                   dic_lent, dic_ff, v_mat)
                                                    # print("VSAV_to_return", VSAV_to_return, "all_ff_waiting", all_ff_waiting)

                                                if not all_ff_waiting and FPT_to_return and (current_station==FPT_to_station) and \
                                                "FPT" in dic_functions[v_mat]:    
                                                    all_ff_waiting = are_all_ff_waiting(ff_existing, current_station, \
                                                                                                   dic_lent, dic_ff, v_mat)
                                                    VSAV_to_return = False
                                                    # print("FPT_to_return", FPT_to_return, "all_ff_waiting", all_ff_waiting)

                                                if not all_ff_waiting and EPA_to_return and (current_station==EPA_to_station) and \
                                                "EPA" in dic_functions[v_mat]:   
                                                    all_ff_waiting = are_all_ff_waiting(ff_existing, current_station, \
                                                                                                   dic_lent, dic_ff, v_mat)
                                                    FPT_to_return = False
                                                    # print("EPA_to_return", EPA_to_return, "all_ff_waiting", all_ff_waiting)

                                                if all_ff_waiting:
                                                    # print("all_ff_waiting", v_mat, current_station)
                                                    ff_existing = dic_lent[current_station][v_mat]
                                                    # print(ff_existing)

                                                else: 
                                                    # print("not all ff waiting")
                                                    ff_existing = [f for f in ff_existing if dic_ff[f] > -1].copy()
                                                    EPA_to_return = False

                                            else: # no vehicle waiting
                                                lent_ff = [ff for v_m in dic_lent.values() for ff_lent in v_m.values() for ff in ff_lent]
                                                ff_not_lent = [num for num in ff_existing if num not in lent_ff]
                                                ff_existing = [f for f in ff_not_lent if dic_ff[f] > -1].copy()

                                            ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)

                                            state = gen_state(veh_depart, idx_role, ff_array, ff_existing, \
                                                              dic_roles, dic_roles_skills, dic_ff, df_skills, \
                                                              coord_x, coord_y, month_sin, month_cos, day_sin, \
                                                              day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size)

                                            if compute and args.train:
                                                l0 = agent.step(old_state, action, reward, state, inter_done)
                                                if l0 is not None:
                                                    loss = l0
                                            else:
                                                loss = 0

                                            action, skill_lvl = agent.act(state, all_ff_waiting, eps)

                                            # if all_ff_waiting:
                                                # print("action", action)

                                            dic_indic, dic_lent, all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff, idx_role, \
                                            degraded = step(action, idx_role, ff_existing, all_ff_waiting, current_station, Z_1, dic_lent, \
                                                            v_mat, dic_ff, VSAV_lent, FPT_lent, EPA_lent, planning, month, day, hour, num_inter, \
                                                            new_required_departure, num_d, list_v, num_role, mandatory, degraded, team_max, \
                                                            all_roles_found, vehicle_found, dic_vehicles, dic_indic, \
                                                            skill_lvl, station_lvl)

                                            # REWARD

                                            reward = compute_reward(dic_indic, dic_indic_old, num_d, dic_tarif)
                                            dic_indic_old = dic_indic.copy()                                            
                                            score += reward
                                            action_num += 1
                                            reward_evo.append([action_num, reward])

                                            old_state = state
                                            compute = True

                                # else: # si aucun véhicule n'a la fonction requise
                                #     print(num_inter, veh_depart, vehicle_to_find, "no vehicule found")

        old_date = date       

        if num_inter % 100 == 0 and required_departure == {0:"RETURN"}:
            rwd_mean = np.mean([row[1] for row in reward_evo[-100:]])
            if args.agent_model == "ppo":
                lr_C = agent.actor_optimizer.param_groups[0]['lr']
                lr_O = agent.critic_optimizer.param_groups[0]['lr']
                lr = 0
            else:
                lr = agent.optimizer.param_groups[0]['lr']
                lr_C = lr_O = 0

            print(f"{num_inter} v_out: {vehicle_out} | rwd_mean: {rwd_mean:.2f} | v1notfroms1: {dic_indic['v1_not_sent_from_s1']} | v3notfroms3: {dic_indic['v3_not_sent_from_s3']} | v_not_found_ls: {dic_indic['v_not_found_in_last_station']} | deg: {dic_indic['v_degraded']} | rupture_ff: {dic_indic["rupture_ff"]}", flush=True)
            print(f"{num_inter} z1_VSAV_sent: {dic_indic['z1_VSAV_sent']} | z1_FPT_sent: {dic_indic['z1_FPT_sent']} | z1_EPA_sent: {dic_indic['z1_EPA_sent']} | VSAV_disp: {VSAV_disp} | FPT_disp: {FPT_disp} | EPA_disp: {EPA_disp} |", flush=True)

            dic_delta = {key:(dic_indic[key] - dic_indic_100[key]) for key in dic_indic}




        if args.eps_start > 0 and num_inter % eps_update == 0 and required_departure == {0:"RETURN"}:
            eps = eps * 0.99**d
            eps = max(0.05, eps)
            d+=1
        # if num_inter % 99 == 0:
        #     clear_output(wait=True)

        if num_inter % 10000 == 0 and args.train and required_departure == {0:"RETURN"}:   
            torch.save(agent.qnetwork_local.state_dict(), args.model_name)
            print(num_inter, "Agent saved as", args.model_name, flush=True)





if __name__ == "__main__":
    main()
