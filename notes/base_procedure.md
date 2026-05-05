# Original Data-generating mechanism

Data were simulated under the Joint Temporal model structure with the following procedure:

1. We simulated random effects using $\xi_i \sim \mathcal{N}\left(0, \sigma^2_\xi\right)$ and $\tau_i \sim \mathcal{N}\left(t_0, \sigma^2_\tau\right)$.
2. We modelled the age at first visit $t_{i,0}$ as $t_{i,0} = \tau_i + \delta_{f_i}$ with $\delta_{f_i} \sim \mathcal{N}\left(\bar{\delta_f}, \sigma^2_{\delta_f}\right)$.
3. We set a time of follow-up per patient $T_{f_i}$, with $T_{f_i} \sim \mathcal{N}\left(\bar{T_f}, \sigma^2_{T_f}\right)$ and a time between two visits $\delta v_{i,j} \sim \mathcal{N}\left(\bar{\delta_v}, \sigma^2_{\delta_v}\right)$ to simulate $n_i$ visits until $t_{n_i} \leq t_{i,0} + T_{f_i} < t_{n_i+1}$.
4. We set the value of the outcome at each visit using a beta distribution of concentration $p$ and mode $\gamma_0(\psi_i(t_{i,j}))$ so that $y_{i,j} \sim \mathcal{B}\left(\gamma_0(\psi_i(t_{i,j})), p\right)$.
5. For each patient, we simulated the event $T_{e_i}$ through a Weibull distribution using $T_{e_i} \sim e^{-\xi_i}W(\nu, \rho) + \tau_i$.
6. We considered that the event stopped the follow-up and that the follow-up censored the event. Thus all the visits after the event were censored: $t_{i,j} > T_{e_i}$ and events after the last visit were censored: $t_{i,\max(j)} < T_{e_i}$.

# Actual Implemented Procedure 

The procedure above is faithful to the model generative story, but generates many patients with only one visit and a follow-up duration of exactly zero: if the Weibull event time $T_{e_i}$ falls before the first planned visit $t_{i,0}$, step 6 removes all visits, leaving the patient with at most a single data point. This mechanically inflates the mass near zero follow-up.

To remove this artefact while keeping the same marginal distributions, the implementation reverses the order in which the event and the visit schedule are constructed:

1. **Sample individual parameters** as before: $\xi_i \sim \mathcal{N}(0, \sigma^2_\xi)$ and $\tau_i \sim \mathcal{N}(t_0, \sigma^2_\tau)$.

2. **Sample the event time first**, before any visit is placed. For each competing event $k$, draw
$$T_{e_i}^{(k)} \sim e^{-\xi_i} W(\nu_k, \rho_k) + \tau_i,$$
and take the first-occurring event: $T_{e_i} = \min_k T_{e_i}^{(k)}$.

3. **Sample the study window independently** of the event:
   - First-visit offset: $\delta_{f_i} \sim \mathcal{N}(\bar{\delta}_f, \sigma^2_{\delta_f})$.
   - Follow-up duration: $T_{f_i} \sim |\mathcal{N}(\bar{T}_f, \sigma^2_{T_f})|$ (absolute value to ensure positivity).
   - Natural study-end: $S_i = \tau_i + \delta_{f_i} + T_{f_i}$.

4. **Set the anchor** as the end of the effective observation window:
$$A_i = \begin{cases} T_{e_i} & \text{if } T_{e_i} \leq S_i \quad \text{(event observed)}, \\ S_i & \text{if } T_{e_i} > S_i \quad \text{(event censored)}. \end{cases}$$

5. **Build the visit schedule forward from** $A_i - T_{f_i}$: starting at $t_{i,0} = A_i - T_{f_i}$, successive inter-visit gaps $\delta v_{i,j} \sim \mathcal{N}(\bar{\delta}_v, \sigma^2_{\delta_v})$ are added until the next visit would exceed $A_i$. Only visits $t_{i,j} \leq A_i$ are retained.

6. **Set longitudinal outcomes** as before with beta-distributed noise around the model trajectory $\psi_i(t_{i,j})$.

**Why this solves the peaks-at-0 problem.** In the original procedure, a very early event time $T_{e_i}$ wipes out the entire visit schedule built in step 3, producing patients with zero follow-up. In the anchor-based procedure, the visit window is constructed *relative to the anchor*, so its duration is always $T_{f_i}$ regardless of whether the event is early or late. Every patient therefore has at least one baseline visit and a follow-up spread over $T_{f_i}$ years, which matches the empirical distribution of the training cohort far more closely.
