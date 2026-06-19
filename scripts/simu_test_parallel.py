import argparse
import os
import pickle
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch

from leaspy.io.data import Data
from leaspy.models import JointModel

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants (never change between runs) ─────────────────────────────────────
NB_EVENTS = 1
MODELS_DIR = os.path.join(".", "models")
SAVE_DIR   = os.path.join("/network/iss/aramis/users/jv.martini", "output")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Joint-model simulation study")
    p.add_argument("--model",   type=str, required=True,
                   help="Model filename (with or without .json) inside ./models/")
    p.add_argument("--N",       type=int, required=True, help="Patients per simulation")
    p.add_argument("--M",       type=int, required=True, help="Number of simulations")
    p.add_argument("--N-iter",  type=int, required=True, help="MCMC-SAEM iterations")
    p.add_argument("--N-perso", type=int, required=True, help="Personalisation iterations")
    p.add_argument("--task-id", type=int, default=None,
                   help="Simulation index (0…M-1). Falls back to $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--aggregate", action="store_true",
                   help="Combine per-task results into one pickle.")
    return p.parse_args()


# ── Path helpers ──────────────────────────────────────────────────────────────
def _model_path(model_name):
    """Return the full path to the model JSON file."""
    stem = model_name.replace(".json", "")
    return os.path.join(MODELS_DIR, stem + ".json")


def _base_stem(model_name, N, M, N_ITER, N_PERSO):
    stem = os.path.basename(model_name).replace(".json", "")
    return f"{stem}_N={N}_M={M}_Niter={N_ITER}_Nperso={N_PERSO}"


def _task_path(model_name, N, M, N_ITER, N_PERSO, m):
    ndigits = len(str(M - 1))
    return os.path.join(
        SAVE_DIR,
        f"{_base_stem(model_name, N, M, N_ITER, N_PERSO)}_task{m:0{ndigits}d}.pkl",
    )


def _aggregate_path(model_name, N, M, N_ITER, N_PERSO):
    return os.path.join(
        SAVE_DIR,
        f"{_base_stem(model_name, N, M, N_ITER, N_PERSO)}.pkl",
    )


# ── Atomic pickle I/O ────────────────────────────────────────────────────────
def atomic_pickle_dump(obj, path):
    """Write pickle via a temp file + atomic rename — no truncated files."""
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # atomic on the same filesystem
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def safe_pickle_load(path):
    """Load a pickle, returning None (+ a warning) if it is corrupted."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"corrupt/truncated file {path}: {exc}")
        return None


# ── Single simulation ─────────────────────────────────────────────────────────
def run_single(m, model_name, N, M, N_ITER, N_PERSO):
    ref_model_path = _model_path(model_name)
    out_path = _task_path(model_name, N, M, N_ITER, N_PERSO, m)

    # Skip only if the existing file is actually readable
    if os.path.exists(out_path):
        if safe_pickle_load(out_path) is not None:
            print(f"[m={m}] Already done — skipping ({out_path})")
            return
        print(f"[m={m}] Existing file corrupted — re-running")

    # Load reference model (per-task, so no cross-process state sharing)
    ref_model = JointModel.load(ref_model_path)
    features  = ref_model.features
    
    # Extract observation model configuration to ensure new models match reference
    obs_model_y = ref_model.obs_models[0]
    obs_model_name = obs_model_y.to_string()

    visit_params = {
        "patient_number": N,
        "visit_type": "random",
        "first_visit_mean": -2.0,
        "first_visit_std":  1.0,
        "time_follow_up_mean": 6.0,
        "time_follow_up_std":  2.0,
        "distance_visit_mean": 0.083,
        "distance_visit_std":  0.042,
        "min_spacing_between_visits": 0.05,
    }

    print(f"[m={m}] Simulating (N={N}, model={model_name}) …")
    np.random.seed(m)
    torch.manual_seed(m)

    _n_digits = len(str(N - 1))

    sim_result = ref_model.simulate(
        algorithm="joint_simulate",
        features=features,
        visit_parameters=visit_params,
    )
    sim_data = sim_result.data

    # Zero-pad IDs so lexicographic == numeric ordering
    _df_sim = sim_data.to_dataframe()
    _id_map = {old: old.zfill(_n_digits) for old in _df_sim["ID"].unique()}
    _df_sim["ID"] = _df_sim["ID"].map(_id_map)
    sim_data = Data.from_dataframe(_df_sim, "joint",
                                   factory_kws={"nb_events": NB_EVENTS})

    ip_sim     = sim_result.individual_parameters
    true_ip_df = ip_sim if isinstance(ip_sim, pd.DataFrame) else pd.DataFrame(ip_sim)
    true_ip_df.index = true_ip_df.index.map(lambda x: x.zfill(_n_digits))

    # Oracle: personalise with the TRUE reference model (known θ).
    ref_ip_df = None
    try:
        ref_ip    = ref_model.personalize(sim_data, "mean_posterior",
                                          seed=3000 + m, n_iter=N_PERSO,
                                          progress_bar=False)
        ref_ip_df = ref_ip.to_dataframe()
    except Exception as exc:
        print(f"[m={m}] Oracle personalisation failed: {exc}")

    # Fit
    new_model = JointModel(
        name=f"m{m}",
        nb_events=ref_model.nb_events,
        dimension=ref_model.dimension,
        source_dimension=ref_model.source_dimension,
        obs_models=obs_model_name,
    )
    try:
        new_model.fit(sim_data, "mcmc_saem",
                      seed=1000 + m, n_iter=N_ITER, progress_bar=True)
    except Exception as exc:
        print(f"[m={m}] Fit failed: {exc}")
        return

    est_params = {
        k: v.detach().cpu().numpy().copy()
        for k, v in new_model.parameters.items()
    }

    # Personalise
    est_ip_df = None
    try:
        est_ip    = new_model.personalize(sim_data, "mean_posterior",
                                          seed=2000 + m, n_iter=N_PERSO,
                                          progress_bar=False)
        est_ip_df = est_ip.to_dataframe()
    except Exception as exc:
        print(f"[m={m}] Personalisation failed: {exc}")

    atomic_pickle_dump(
        {"pop_params": est_params, "true_ip": true_ip_df,
         "ref_ip": ref_ip_df, "est_ip": est_ip_df},
        out_path,
    )
    print(f"[m={m}] Saved → {out_path}")


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(model_name, N, M, N_ITER, N_PERSO):
    agg_path = _aggregate_path(model_name, N, M, N_ITER, N_PERSO)
    results  = {"pop_params": [], "true_ips": [], "ref_ips": [], "est_ips": []}
    n_ok = 0

    for m in range(M):
        path = _task_path(model_name, N, M, N_ITER, N_PERSO, m)
        if not os.path.exists(path):
            print(f"[aggregate] Missing m={m}: {path}")
            results["pop_params"].append(None)
            results["true_ips"].append(None)
            results["ref_ips"].append(None)
            results["est_ips"].append(None)
            continue

        r = safe_pickle_load(path)
        if r is None:
            results["pop_params"].append(None)
            results["true_ips"].append(None)
            results["ref_ips"].append(None)
            results["est_ips"].append(None)
            continue

        results["pop_params"].append(r["pop_params"])
        results["true_ips"].append(r["true_ip"])
        results["ref_ips"].append(r.get("ref_ip"))
        results["est_ips"].append(r["est_ip"])
        n_ok += 1

    atomic_pickle_dump(results, agg_path)
    print(f"[aggregate] {n_ok}/{M} results → {agg_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args       = parse_args()
    MODEL_NAME = args.model
    N          = args.N
    M          = args.M
    N_ITER     = args.N_iter
    N_PERSO    = args.N_perso

    # Verify model file exists early
    ref_model_path = _model_path(MODEL_NAME)
    if not os.path.isfile(ref_model_path):
        raise FileNotFoundError(f"Model file not found: {ref_model_path}")

    if args.aggregate:
        aggregate(MODEL_NAME, N, M, N_ITER, N_PERSO)
    else:
        task_id = args.task_id
        if task_id is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env is not None:
                task_id = int(env)

        if task_id is not None:
            # ── HPC array-task mode ───────────────────────────────────────
            if task_id == 0:
                ref = JointModel.load(ref_model_path)
                print(f"True parameters θ (model: {MODEL_NAME}):")
                for k, v in ref.parameters.items():
                    print(f"  {k:25s}: "
                          f"{np.atleast_1d(v.detach().cpu().numpy()).tolist()}")
            run_single(task_id, MODEL_NAME, N, M, N_ITER, N_PERSO)
        else:
            # ── Local sequential fallback ─────────────────────────────────
            ref = JointModel.load(ref_model_path)
            print(f"True parameters θ (model: {MODEL_NAME}):")
            for k, v in ref.parameters.items():
                print(f"  {k:25s}: "
                      f"{np.atleast_1d(v.detach().cpu().numpy()).tolist()}")
            for m_idx in range(M):
                run_single(m_idx, MODEL_NAME, N, M, N_ITER, N_PERSO)
            aggregate(MODEL_NAME, N, M, N_ITER, N_PERSO)