# Original Data-generating mechanism

Data were simulated under the Joint Temporal model structure with the following procedure:

1. We simulated random effects using $\xi_i \sim \mathcal{N}\left(0, \sigma^2_\xi\right)$ and $\tau_i \sim \mathcal{N}\left(t_0, \sigma^2_\tau\right)$.
2. We modelled the age at first visit $t_{i,0}$ as $t_{i,0} = \tau_i + \delta_{f_i}$ with $\delta_{f_i} \sim \mathcal{N}\left(\bar{\delta_f}, \sigma^2_{\delta_f}\right)$.
3. We set a time of follow-up per patient $T_{f_i}$, with $T_{f_i} \sim \mathcal{N}\left(\bar{T_f}, \sigma^2_{T_f}\right)$ and a time between two visits $\delta v_{i,j} \sim \mathcal{N}\left(\bar{\delta_v}, \sigma^2_{\delta_v}\right)$ to simulate $n_i$ visits until $t_{n_i} \leq t_{i,0} + T_{f_i} < t_{n_i+1}$.
4. We set the value of the outcome at each visit using a beta distribution of concentration $p$ and mode $\gamma_0(\psi_i(t_{i,j}))$ so that $y_{i,j} \sim \mathcal{B}\left(\gamma_0(\psi_i(t_{i,j})), p\right)$.
5. For each patient, we simulated the event $T_{e_i}$ through a Weibull distribution using $T_{e_i} \sim e^{-\xi_i}W(\nu, \rho) + \tau_i$.
6. We considered that the event stopped the follow-up and that the follow-up censored the event. Thus all the visits after the event were censored: $t_{i,j} > T_{e_i}$ and events after the last visit were censored: $t_{i,\max(j)} < T_{e_i}$.
