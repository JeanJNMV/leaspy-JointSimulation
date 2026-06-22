## Parameter Estimation Procedure

### 1. `first_visit_mean` and `first_visit_std`

These describe the offset $\delta_{f_i}$ relative to $\tau_i$ (the individual time-shift, a model parameter). Since $\tau_i$ is learnt by the model, you need the **residual** first-visit offset after removing the model's time-reparametrization.

**In practice**, if you don't have access to the inferred $\tau_i$ values yet, use the raw first-visit ages as a proxy:

$$\hat{\delta}_{f_i} \approx t_{i,0} - \bar{t}_0$$

Compute: `first_visit_mean = mean(first visit TIME per ID) - global mean`, `first_visit_std = std(first visit TIME per ID)`. Once the model is fitted, replace $t_{i,0}$ with $t_{i,0} - \hat{\tau}_i$ for each patient and recompute.

---

### 2. `time_follow_up_mean` and `time_follow_up_std`

$T_{f_i}$ is the total duration of follow-up per patient, **before censoring by the event**. You want the *intended* follow-up window, not the truncated one.

- For **censored patients** (`EVENT_BOOL = 0`): the last visit is the end of follow-up, so $T_{f_i} = t_{i, \max(j)} - t_{i,0}$.
- For **event patients** (`EVENT_BOOL = 1`): the observed follow-up is truncated by the event. The true $T_{f_i}$ is unobserved, but you know it satisfies $T_{f_i} \geq t_{i,\max(j)} - t_{i,0}$. Use the censored patients to fit the distribution, or apply a **reverse KM estimator** on $T_{f_i}$ treating event patients as right-censored at their last visit time. A simple approximation: pool all patients and use last-visit duration, accepting a slight downward bias.

$$\bar{T}_f = \text{mean}(t_{i,\max} - t_{i,0}), \quad \sigma_{T_f} = \text{std}(t_{i,\max} - t_{i,0})$$

---

### 3. `distance_visit_mean` and `distance_visit_std`

These are the inter-visit gaps $\delta v_{i,j} = t_{i,j+1} - t_{i,j}$.

Compute all consecutive time differences across all patients and all visit pairs:

$$\bar{\delta}_v = \text{mean}_{i,j}(t_{i,j+1} - t_{i,j}), \quad \sigma_{\delta_v} = \text{std}_{i,j}(t_{i,j+1} - t_{i,j})$$

Exclude the last gap before event/censoring if you suspect it is irregular (e.g., emergency visit). Check the distribution visually — if it is skewed, a log-normal may fit better than a Gaussian, but the parameters here assume Gaussian so use mean/std directly.

---

### 4. `min_spacing_between_visits`

Set this to the **empirical minimum** inter-visit gap observed in the dataset, rounded down conservatively:

$$\min\_spacing = \min_{i,j}(t_{i,j+1} - t_{i,j})$$

This acts as a hard floor in simulation, so it should reflect the clinical minimum (e.g. protocol-mandated spacing).

---

### 5. `patient_number`

Set to the number of unique IDs in your dataset for a like-for-like simulation. Increase it for stability studies.

---

### Summary Table

| Parameter | Estimator | Notes |
|---|---|---|
| `first_visit_mean` | $\text{mean}(t_{i,0} - \hat{\tau}_i)$ | Use raw mean if $\hat\tau_i$ unavailable |
| `first_visit_std` | $\text{std}(t_{i,0} - \hat{\tau}_i)$ | Same |
| `time_follow_up_mean` | $\text{mean}(t_{i,\max} - t_{i,0})$ | Fit on censored patients, or use KM for all |
| `time_follow_up_std` | $\text{std}(t_{i,\max} - t_{i,0})$ | Same caveat |
| `distance_visit_mean` | $\text{mean}_{i,j}(\Delta t_{i,j})$ | All consecutive gaps |
| `distance_visit_std` | $\text{std}_{i,j}(\Delta t_{i,j})$ | All consecutive gaps |
| `min_spacing_between_visits` | $\min_{i,j}(\Delta t_{i,j})$ | Round to clinical floor |

---

