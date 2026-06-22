# The Beta distribution problem in `joint_simulate`

## What was wrong

The `_generate_dataset` method in `joint_simulate.py` added observation noise by sampling from a **Beta distribution**, using the model's `noise_std` parameter as a standard deviation and converting it to a variance:

```python
var = noise_std ** 2
alpha = mu * (mu*(1-mu)/var - 1)
beta  = (1-mu) * (mu*(1-mu)/var - 1)
observation = beta.rvs(alpha, beta)
```

This is the correct approach for a **LogisticModel**, which is trained with a `gaussian-diagonal` observation model whose per-feature `noise_std` values are small (typically 0.08–0.17 for the PULSE logistic model). With such small variances the Beta shape parameters are large (α, β ≫ 1), producing a well-concentrated, bell-shaped distribution around `mu`.

The **JointModel**, however, is trained with a `gaussian-scalar` observation model: a single shared Gaussian noise for all features, with `noise_std` representing a Gaussian standard deviation — not a Beta variance.

For the PULSE joint model (`PULSE_JOINT_100_ALSFRS_BMI_VC_MUSC_NFL_SNIP_10.json`):

```
noise_std = 0.391  →  var = 0.391² ≈ 0.153
```

### Why this is catastrophic for the Beta distribution

A Beta distribution on (0, 1) has a hard constraint on its variance:

$$\text{Var}[\text{Beta}(\alpha, \beta)] = \frac{\mu(1-\mu)}{\alpha+\beta+1} < \mu(1-\mu)$$

The maximum achievable variance is $\mu(1-\mu)$, which is at most 0.25 (at $\mu = 0.5$). With `var ≈ 0.153`, the variance is already close to or exceeds this maximum for any $\mu$ away from 0.5:

| $\mu$ | max var $= \mu(1-\mu)$ | clamped var | $\alpha$ | $\beta$ | shape |
|-------|----------------------|-------------|----------|---------|-------|
| 0.2   | 0.160                | 0.153       | 0.009    | 0.037   | **U-shaped** |
| 0.3   | 0.210                | 0.153       | 0.112    | 0.262   | **U-shaped** |
| 0.5   | 0.250                | 0.153       | 0.318    | 0.318   | **U-shaped** |
| 0.7   | 0.210                | 0.153       | 0.262    | 0.112   | **U-shaped** |

When both $\alpha < 1$ and $\beta < 1$, the Beta distribution is **U-shaped**: probability mass concentrates at 0 and 1. Every simulated observation snapped to either 0 or 1, regardless of the true trajectory value — which is exactly what was observed in the plots.

For comparison, the LogisticModel with `noise_std[0] = 0.0795` (`var ≈ 0.006`) gives α ≈ 5–23 and β ≈ 10–23 → bell-shaped, centred on the trajectory.


