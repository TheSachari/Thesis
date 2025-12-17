# Preprocessing

Running ```python3 preprocess.py``` for the first time will create the following folders :

["Data_preprocessed", "Data_trained", "Data_sampled", "Data_environment", "SVG_model", "Plots", "Reward_weights"]

Preprocesses the raw data from the ```./Data``` folder.

# Train

```train.py --lr 0.0025 --layers 1024 --num_timesteps 1000 --is_y_cond --save_as agent_name ```

Trains the model. 

# Sample

```train.py --lr 0.0025 --layers 1024 --num_timesteps 1000 --is_y_cond --save_as agent_name ```

Samples new interventions from a trained model. 

# Explainability

```explainability.py```

Precomputes rare skills for explainability.

# Simulation


```simulation_start.py --dataset df_pc_real.pkl --start 1 --end 202 --constraint_factor 3 --reward_weights reward_weights.json --save_metrics_as metrics_name```

Runs the simulation.

# Agent

```agent_run.py --model_name agent_name --agent_model fqf --hyper_params hyper_params.json --reward_weights rw_weights.json --dataset df_pc.pkl --start 1 --end 63696 --constraint_factor_veh 1 --constraint_factor_ff 1 --save_metrics_as metrics```

Trains an agent with ```--train```. \
Loads training weights in training mode with ```--load```. \
Otherwise, runs the agent ```agent_name``` in test mode.










