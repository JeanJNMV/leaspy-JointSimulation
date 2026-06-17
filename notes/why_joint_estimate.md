# Why Use `estimate` Instead of Sampling from the Weibull Directly?

## Background: Two Ways to Simulate Event Times

The joint model links longitudinal trajectories to survival outcomes via a shared latent disease age
$\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0$. The survival sub-model is a Weibull distribution applied to
this latent age. There are two conceptually distinct ways to simulate an event time for a patient:

1. **Direct Weibull sampling** (original algorithm): draw $T_{e_i}$ analytically from the Weibull
   formula implied by the patient's individual parameters.
2. **`estimate`-based sampling** (implementation choice): evaluate the model's own `estimate` method
   at each planned visit time, read off the cumulative distribution function (CDF) of the event, and
   apply inverse-transform sampling on that discrete grid.

---

## The Original Approach: Direct Weibull Sampling

The original simulation algorithm derives event times from a closed-form expression:

$$T_{e_i} \sim e^{-\xi_i}\,\mathcal{W}(\nu, \rho) + \tau_i$$

This is mathematically equivalent to drawing from the Weibull distribution and mapping the result
through the patient's individual parameters $(\xi_i, \tau_i)$. It is fast, exact, and continuous —
the event time can land at any real value.

**However**, it has two important limitations:

- It requires re-deriving the Weibull parameters from model internals (scale $\nu$, shape $\rho$,
  individual accelerations $\xi_i$, and source-to-survival mixing weights $\zeta$ for competing
  events). This duplicates logic that already lives inside the model.
- It does not naturally extend to the competing-events setting, where multiple event types each have
  their own Weibull parameters coupled through the source mixing matrix $\zeta$. Handling this
  correctly requires assembling cause-specific CIFs by hand, which is error-prone and not reusable.

---

## The `estimate`-Based Approach

The `estimate` method is the model's own prediction engine. Given a set of planned visit times and
individual parameters, it evaluates the full trajectory at each visit and returns — in addition to
longitudinal feature values — the survival/event predictions already assembled by the model:

- **Single-event model**: returns the **conditional survival** $S_i(t_{i,j}) / S_i(t_{i,0})$ at
  each visit, conditioned on the patient being alive at the first visit $t_{i,0}$.
- **Competing-events model**: returns the **conditional cause-specific CIF**
  $\mathrm{CIF}_{i,l}(t_{i,j}) / S_i(t_{i,0})$ for each event type $l$, again conditioned on
  survival at $t_{i,0}$.

From these, the total conditional CDF is assembled as:

$$F_i^{\,\text{cond}}(t_{i,j}) =
\begin{cases}
1 - S_i(t_{i,j})\,/\,S_i(t_{i,0}) & \text{(single event)} \\[4pt]
\displaystyle\sum_{l=1}^{L} \mathrm{CIF}_{i,l}(t_{i,j})\,/\,S_i(t_{i,0}) & \text{(competing events)}
\end{cases}$$

Event time sampling then proceeds by **inverse-transform sampling on the discrete visit grid**: draw
$U \sim \mathcal{U}[0,1]$ and set $T_{e_i}$ to the first visit time $t_{i,j}$ at which
$F_i^{\,\text{cond}}(t_{i,j}) \geq U$. If $U$ exceeds the total CDF at the last planned visit, the
patient is right-censored at that visit.

---

## Advantages of the `estimate`-Based Approach

### 1. Single source of truth — no logic duplication

The Weibull-based event probability computation is already implemented (and tested) inside `estimate`.
Using it directly means the simulation reuses the exact same code path as the prediction step.
Any future change to the survival sub-model (e.g., a new parameterization, a bug fix) is
automatically reflected in both prediction and simulation, with no risk of the two drifting apart.

### 2. Automatic generalization to competing events

Extending direct Weibull sampling to $L > 1$ competing events requires assembling cause-specific
CIFs from scratch, correctly handling the source mixing matrix $\zeta$, and ensuring that the
individual CIFs are consistent with the overall survival. With the `estimate` approach, none of this
is needed: `estimate` already returns the per-event conditional CIFs for any $L$, and the
simulation code is identical regardless of whether $L = 1$ or $L > 1$.

### 3. Self-consistency between generation and likelihood

In a simulation study, data are generated under the model and then re-estimated using the same
model. For the evaluation to be valid, the generative process must be exactly consistent with the
likelihood as the model computes it. By routing event-time simulation through `estimate`, the
generated event times are guaranteed to be draws from precisely the distribution that the
likelihood evaluates — not from a hand-coded approximation of it.

### 4. Survival conditioning at the first visit

The conditioning on $S_i(t_{i,0})$ (survival at the first planned visit) is a natural consequence
of using `estimate`: the method already returns quantities conditioned on the patient being alive at
$t_{i,0}$. This is coherent with the observation model, where a patient appears in the dataset only
if they attended at least one visit. Direct Weibull sampling does not enforce this automatically
and would require an explicit rejection or conditioning step.

---

## Trade-offs and Limitations

The choice comes with two notable trade-offs, explicitly acknowledged in the report:

- **Discrete event times**: because the CDF is evaluated only at the planned visit times (not
  continuously), event times are constrained to lie on the visit grid. This discretization is
  coarser than drawing from a continuous Weibull, but is coherent with the fact that events in
  clinical datasets are recorded at visit times anyway.

- **Event cannot precede the first visit**: because sampling is conditioned on survival at
  $t_{i,0}$, the algorithm can never assign an event before the first planned visit. This is a
  deliberate design choice that matches the clinical reality of the training data.

---

## Summary

| Criterion | Direct Weibull sampling | `estimate`-based sampling |
|---|---|---|
| Requires manual Weibull formula | Yes | No |
| Works out of the box for competing events | No (requires extra code) | Yes |
| Consistent with model likelihood | Only if implemented identically | Yes, by construction |
| Survival conditioned on first visit | Not automatic | Yes, built-in |
| Continuous event times | Yes | No (discrete grid) |
| Code reuse / maintenance | Duplicates model logic | Single source of truth |

In short, using `estimate` trades a small amount of temporal precision (continuous vs. discrete
event times) for a large gain in correctness, consistency, and generality.
