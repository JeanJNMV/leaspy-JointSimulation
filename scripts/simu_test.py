import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, beta as _beta_dist

from leaspy.io.data import Data
from leaspy.models import JointModel
from leaspy.datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)

M          = 20 # Number of simulations per condition
N          = 200 # Number of samples per simulation 
N_ITER     = 10000   
N_PERSO    = 1000   
NB_EVENTS  = 1

visit_params = {
    "patient_number": N,
    "visit_type": "random",
    "first_visit_mean": -2.0,    # start 2 years BEFORE disease reference time τ
    "first_visit_std":  1.0,
    "time_follow_up_mean": 6.0,   
    "time_follow_up_std":  2.0,
    "distance_visit_mean": 0.083,
    "distance_visit_std":  0.042,
    "min_spacing_between_visits": 0.05,
} 

REF_MODEL_PATH = os.path.join(".", "models", "PULSE_JOINT_100_ALSFRS_BMI_VC_MUSC_NFL_SNIP_10.json")
ref_model = JointModel.load(REF_MODEL_PATH)

FEATURES  = ref_model.features
SAVE_PATH = os.path.join(".", "output", os.path.basename(REF_MODEL_PATH).replace(".json", f"_N={N}_M={M}_Niter={N_ITER}_Nperso={N_PERSO}.pkl"))

theta_true = {
    k: v.detach().cpu().numpy().copy()
    for k, v in ref_model.parameters.items()
}

print("True parameters theta (reference model):")
for k, v in theta_true.items():
    print(f"  {k:25s}: {np.atleast_1d(v).tolist()}")
    

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded {len(results['pop_params'])} results from {SAVE_PATH}")
else:
    results = {
        "pop_params": [],   # list of M dicts {param_name: np.ndarray}
        "true_ips":   [],   # list of M DataFrames with columns xi, tau (from simulation)
        "est_ips":    [],   # list of M DataFrames with columns xi, tau (from personalisation)
    }

    # Number of digits needed so zero-padded IDs sort lexicographically == numerically.
    # e.g. N=50 → 2 digits: "00","01",...,"49"  (avoids "0","1","10","11",... ordering)
    _n_digits = len(str(N - 1))

    for m in range(M):
        print(f"Simulation {m} starting.")
        # Simulate a dataset with the reference model
        np.random.seed(m)
        torch.manual_seed(m)

        sim_result = ref_model.simulate(
            algorithm="joint_simulate",
            features=FEATURES,
            visit_parameters=visit_params,
        )
        sim_data = sim_result.data

        # Rename IDs to zero-padded strings so that groupby("ID").min() inside
        # JointModel's Weibull initialisation produces the same patient order as
        # dataset.event_time (which uses insertion order).
        _df_sim = sim_data.to_dataframe()
        _id_map = {old: old.zfill(_n_digits) for old in _df_sim["ID"].unique()}
        _df_sim["ID"] = _df_sim["ID"].map(_id_map)
        sim_data = Data.from_dataframe(_df_sim, "joint", factory_kws={"nb_events": NB_EVENTS})

        # individual_parameters from simulate is a DataFrame [xi, tau, ...]
        ip_sim = sim_result.individual_parameters
        true_ip_df = ip_sim if isinstance(ip_sim, pd.DataFrame) else pd.DataFrame(ip_sim)
        true_ip_df.index = true_ip_df.index.map(lambda x: x.zfill(_n_digits))

        # Fit a new model on the simulated data to recover population parameters
        new_model = JointModel(name=f"m{m}", nb_events=NB_EVENTS)
        try:
            new_model.fit(
                sim_data, "mcmc_saem",
                seed=1000 + m, n_iter=N_ITER, progress_bar=True,
            )
        except Exception as exc:
            print(f"  [m={m}] Fit failed: {exc}")
            continue
        est_params = {
            k: v.detach().cpu().numpy().copy()
            for k, v in new_model.parameters.items()
        }
        results["pop_params"].append(est_params)

        # Personalise the new model to recover individual parameters and compare
        try:
            est_ip = new_model.personalize(
                sim_data, "mean_posterior",
                seed=2000 + m, n_iter=N_PERSO, progress_bar=False,
            )
            results["est_ips"].append(est_ip.to_dataframe())
            results["true_ips"].append(true_ip_df)
        except Exception as exc:
            print(f"  [m={m}] Personalisation failed: {exc}")
            results["est_ips"].append(None)
            results["true_ips"].append(None)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {SAVE_PATH}")
