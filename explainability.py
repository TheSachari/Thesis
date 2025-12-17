import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collective_functions import *
tqdm.pandas()



def get_dic_rare_skills(rare_skills, ff_array):
    if rare_skills.size == 0:
        return {i: [] for i in range(ff_array.shape[0])}
    sub = ff_array[:, rare_skills]   

    return {i: rare_skills[np.flatnonzero(sub[i])].astype(int) for i in range(ff_array.shape[0])}

def get_related_rows_in_time(idx, date, pdd, df, top_n = 5, n_hours=2):

    limit_time = date + timedelta(hours=n_hours)
    target_villes = pdd[:top_n+1]
    selected_skills = []

    for j in range(idx + 1, len(df)):
        t = df.at[j, "date"]
        if t > limit_time:  # on sort dès qu'on dépasse la fenêtre
            break
        if df.at[j, "PDD"]:
            first_city = df.at[j, "PDD"][0]
            if first_city in target_villes:
                selected_skills.extend(df.at[j, "rare_skills_required"])

    return np.unique(np.concatenate(selected_skills, dtype=int)) if selected_skills else np.array([], dtype=int)

def get_skill_counts(date, df_skills):

    skills_updated = update_skills(df_skills, date)
    counts = np.count_nonzero(skills_updated, axis=0)
    return counts

def get_all(date, required_departure, df_skills, rarity, dic_roles, dic_roles_skills):
    counts = np.count_nonzero(update_skills(df_skills, date), axis=0)
    return get_rare_skills_from_dep(required_departure, counts, rarity, dic_roles, dic_roles_skills)

def get_rare_skills_from_dep(required_departure, counts, rarity, dic_roles, dic_roles_skills):
    list_veh = [val for lst in required_departure.values() for val in lst]
    required_roles = set(val for d in [dic_roles[v] for v in list_veh] for val in d.values())
    return np.unique(np.concatenate([get_rare_skills(counts, dic_roles_skills[role]) for role in required_roles]))

def get_rare_skills(rare_skills, ff_skills, rarity = 50, skill_lvl_gt = 0):
    pos_in_b = ff_skills > skill_lvl_gt                   
    a_lt50   = rare_skills < rarity                    
    match    = pos_in_b & a_lt50[None,:] 
    any_match_per_row  = match.any(axis=1)                     
    all_pos_ok_per_row = (~pos_in_b | a_lt50[None,:]).all(axis=1)
    return np.unique(np.concatenate([np.where(row)[0] for row in match]))

def get_all_planed_ff(planning, month, day, hour, df_stations):
    return np.concatenate([planning[c][month][day][hour]["planned"] for c in df_stations["Nom"].values])

def get_rare_skills_from_planed_ff(date, month, day, hour, planning, df_stations, df_skills, rarity):
    array_of_mats = get_all_planed_ff(planning, month, day, hour, df_stations)
    df_skills_filtered = df_skills.loc[array_of_mats]
    updated_skills_filtered = update_skills(df_skills_filtered, date)
    all_current_ff_skills = np.count_nonzero(updated_skills_filtered, axis=0)
    return  np.where(all_current_ff_skills < rarity)

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Environment params")
    parser.add_argument("--rarity", type=int=, help="rarity threshold for skills")
    parser.add_argument("--n_hours", type=int=, help="time window to consider")
    parser.add_argument("--top_n", type=str, help="nearest stations to consider")
    args = parser.parse_args()
    
    os.chdir("./Data_environment/")
    df_pc = pd.read_pickle("df_pc_real_prob.pkl")
    df_stations = pd.read_pickle("df_stations.pkl")
    df_skills = pd.read_pickle("df_skills.pkl")
    planning = pd.read_pickle("planning.pkl")

    df_pc = df_pc[df_pc["departure"] != {0: 'RETURN'}]
    
    df_pc["rare_skills_required"] = df_pc.progress_apply(lambda row: get_rare_skills_from_planed_ff(row["date"], 
                                                                                                    row["Month"], 
                                                                                                    row["Day"], 
                                                                                                    row["Hour"], 
                                                                                                    planning, 
                                                                                                    df_stations, 
                                                                                                    df_skills, 
                                                                                                    args.rarity),
                                                         axis=1)

    df_pc.to_pickle("df_pc_rare_skills.pkl")
    
    df_pc_rs = pd.read_pickle("df_pc_rare_skills.pkl")
    df_pc_real = pd.read_pickle("df_pc_real_prob.pkl")
    
    df_pc_real["rare_skills_required"] = df_pc_real.index.map(lambda _: np.array([], dtype=int))
    df_pc_real.update(df_pc_rs["rare_skills_required"])
    df_pc_real.to_pickle("df_pc_prob_rare_skills_merged.pkl")
    
    os.chdir('../')

if __name__ == "__main__":

    main()

