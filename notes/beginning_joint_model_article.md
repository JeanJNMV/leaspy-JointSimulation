# 1 Introduction

Neurodegenerative disorders are an important burden for the healthcare system ([Zahra et al., 2020]). The heterogeneity of the progression of these diseases is a major challenge in the development of effective therapies, as in Alzheimer's Disease ([Duara and Barker, 2022]) or Amyotrophic Lateral Sclerosis (ALS) ([Beghi et al., 2007]). With the increasing number of large clinical databases, the development of disease progression models has helped in understanding this heterogeneity better. Two main data types can be subjected to modelling: longitudinal data such as repeated measures of clinical scores or biomarkers; or survival data, with the occurrences of events like death or surgical intervention. As for ALS, the U.S. Food and Drug Administration might demand assessing treatment efficacy using both types of outcomes ([FDA, 2019]).

Data available for neurodegenerative disease is often sparse and covers only parts of the progression, which emphasises the need to realign different patients' ages depending on their disease stages to extract a full typical timeline of the disease ([Young et al., 2024]). The date of the first symptom is often used to do so, even if it does not capture the difference in terms of speed of progression. But in such disease the underlying biological processes start earlier than the first specific symptoms of the disease, making the first symptoms reported by the patient not representative of the disease onset and very subjective. For instance, in Alzheimer's Disease, the progression of amyloid and neurodegeneration biomarkers starts before the manifestation of the first clinical symptoms ([Jack et al., 2010]). Similarly in ALS, changes in metabolism start before a significant weight loss and the first motor symptoms ([Peter et al., 2017]), which again complicates the establishment of reliable reference time points for monitoring disease progression. Thus two challenges are faced to extract a full typical progression of these diseases: mapping the disease onset and the speed of progression of the patients.

Survival and longitudinal data are often associated with the same biological processes. In such a case, modelling both data together results in more precise estimates and improves inference ([Lu et al., 2023]). Joint models enable us to do so. They are composed of three parts: a model for the longitudinal data, a model for the survival data and a linkage, often a shared latent variable, which captures the association between the two types of data.

Classical survival models are hazard-based regression models ([Rubio et al., 2019]). The most used is the Cox Proportional Hazard (Cox) model ([Cox, 1972]), with an effect on the hazard scale. Its main interests are that it does not require estimating the baseline hazard function when used alone and that it is easy to interpret due to the proportional impact of covariates on the baseline hazard. Nevertheless, the baseline hazard needs to be estimated for joint models ([Rizopoulos, 2012]) and the proportional assumption is often violated on long follow-up. Another family of models is the Accelerated Failure Time (AFT) model ([Kalbfleisch and Prentice, 2002]). In these models, covariates directly affect survival time, but modelling the hazard function is mandatory, making it less used for survival analysis alone. Besides, the results may be harder to interpret.

Classical longitudinal models are mixed-effect models that allow for modelling repeated and correlated observations ([Laird and Ware, 1982]). To capture the heterogeneity of the patient progression, in addition to population parameters, named fixed effects, they have individual parameters, named random effects. The most common type of Non-Linear Mixed effects are generalised linear Mixed-effect Models (GLMMs), a direct extension of Linear mixed-effects models (LMM), which enables the estimation of non-linear progression ([McCulloch et al., 2008]). Such classical mixed-effect models rely on an empirical disease time axis which limits their temporal resolution to the resolution of the reference time used to index the disease time axis ([Young et al., 2024]). To overcome this issue, longitudinal models that capture the data-driven disease timeline behind the observable data, named data-driven disease progression models, have been developed ([Young et al., 2024]). Among others, disease Progression score models have been developed [Jedynak et al., 2012] and extended ([Bilgel et al., 2019]) as well as models that allow direct integration of knowledge about different stages of diseases ([Raket, 2020]). Finally, some models have explored the creation of a latent disease age ([Li et al., 2017, Iddi et al., 2018, Schiratti et al., 2015, Schiratti, 2017]). A non-linear mixed effect model based on Riemannian geometry was proposed to capture a latent disease age and showed good performances in degenerative disease ([Schiratti, 2017, Marinescu et al., 2019]).

For the linkage structure between survival and longitudinal data, two types of modelling have been proposed: the latent class model ([Proust-Lima et al., 2017, Proust-Lima et al., 2023]) and the shared random effects model ([Rizopoulos, 2016]). Such an approach aggregates similar patients which could help better understand the heterogeneity even though the meaning of the different classes remains unknown. It allows us to calculate the probability of an individual belonging to a particular class but may result in some patients being almost equitably distributed. Shared random effect models are often used to avoid these limits. Nevertheless, they have their own limitation: by including predictors of the longitudinal outcome in the survival model, they usually focus on how longitudinal outcomes impact survival. Whatever their type, all these joint models rely on GLMMs, necessitating the use of reliable reference times, which is not available in our context.

In this paper, we propose a latent age joint model suited for neurodegenerative diseases that overcomes the need for a reliable reference time of the state-of-the-art joint models. To do so, we used an existing longitudinal model with a latent disease age ([Schiratti, 2017]) as the longitudinal sub-model and used its defined latent disease age as the linkage structure. We associated a survival sub-model that estimates a Weibull distribution from the latent disease age.

After introducing the proposed joint model, we validated it on simulated real-like clinical data. We then benchmark the proposed joint model against reference models on real ALS data and show that the proposed approach is better suited in the context of the absence of a reliable reference time point reference time than the state-of-the-art. Finally, we made a clinical application of the Joint Temporal model to present how parameters can be interpreted.

# 2 Model

## 2.1 Generic Framework

### 2.1.1 Data

We consider $N$ patients, associated with longitudinal data: repeated measures of one given outcome $y$. Each patient $i$ is followed for $n_i$ visits. For each visit $j$, we denote $t_{i,j}$ the age at which the outcome is measured, and $y_{i,j}$ the value of the outcome for the patient at this visit.

We assume that we also observe an event $e$, and denote $t^e_i$ the age of the patient when the event is observed. Nevertheless, the event may occur after the follow-up period. In this case, the event is said to be censored, in opposition to observed. To distinguish censored and observed events, a boolean $B^e_i$ is associated with the time of the event $t^e_i$: $B^e_i = 0$ if the event is censored and $B^e_i = 1$ if the event is observed. If the event is censored, the time of the last visit is used as the time of the event ([Leung et al., 1997]).

### 2.1.2 Joint model structure

The objective of joint models is to describe the combination of two types of clinical data: longitudinal data and survival data, with their relationship. Here, the longitudinal process, $\gamma_i(t)$, is the progression of an outcome, measured by $y_{i,j}$ at each time $t_{i,j}$ for each visit $j$ of the patient $i$. The longitudinal process is estimated with a Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma)$ compared to the measure, so that: $y(t_{i,j}) = \gamma(t_{i,j}) + \varepsilon_{i,j}$. The survival process $S_i(t)$, the probability that a patient $i$ experiences the event after age $t$ ($S_i(t) = p(t^e_i > t)$).

To describe the model, we will further use the formalism of mixed-effects models ([Laird and Ware, 1982]). Such models are composed of two types of parameters: parameters that differ from one patient to the other and enable to encapsulate the individual variability, named random effects, and parameters that capture the population specificity and are shared by all the patients, named fixed effects.

## 2.2 The Proposed Joint model

A non-linear mixed-effect model with a latent disease age was first introduced by [Schiratti et al., 2015]. Both the latent age and the modelling of the longitudinal process presented below are extracted from [Schiratti et al., 2015, Schiratti, 2017].

### 2.2.1 The latent disease age: correction of individual variation

The idea of the latent disease age, $\psi_i(t)$, is to map the chronological age of a patient into a latent disease age representative of the disease stage of the patient. Using the formalism described before, it can be written as:

$$\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0 \tag{1}$$

where $e^{\xi_i}$ is the progression rate of patient $i$, $\tau_i$ is its time-shift and $t_0$ is the population estimated reference time.

In the proposed joint model, the idea is to encapsulate all the individual variability of the patient $i$, in the latent disease age $\psi_i$, with random effects. The latent disease age is then used as the link between the longitudinal and survival processes $(\gamma_i(t), S_i(t))$, which are estimated from the latent disease age with the composition of functions that describe only the population $(\gamma_0, S_0)$ and are shared by all patients with fixed effects.

$$\begin{cases} \gamma_i(t) = \gamma_0(\psi_i(t_{i,j})) \\ S_i(t) = S_0(\psi_i(t)) \end{cases}$$

For the survival modelling, instead of using time 0 as a start time, we use the reference time $t_0$ and impose $\forall t < t_0, S_0(t) = 1$. Indeed, $t_0$ is automatically estimated thanks to the visit times and corresponds to a time of a given value of the score that most of the patients experimented with. Thus at that time, patients should not be dead. We have checked that even for the existing longitudinal model most patients were still alive at $t_0$.

Compared to joint models with shared random effects models of ([Rizopoulos, 2016]), where $S_0(\psi_i(t)) = f(\gamma_0(\psi_i(t))$ with $f$ a function of fixed parameters (further detailed in Table 1), the Joint Temporal model directly depends on the latent disease age.

### 2.2.2 Modelling longitudinal process

The modelling of the longitudinal process consists in computing the trajectory from the latent disease age defined in part 2.2.1. We will study a clinical score with curvilinearity, and potential floor or ceiling effects ([Gordon et al., 2010]). Thus a logistic function will be used to model the outcome value from the latent disease age $\psi_i(t)$. It is parametrized as follows:

$$\gamma_0(\psi_i(t)) = \left(1 + g \times \exp\!\left(-v_0\frac{(g+1)^2}{g}(\psi_i(t) - t_0)\right)\right)^{-1} \tag{2}$$

where $t_0$ is the population estimated reference time defined in 2.2.1, $v_0$ is the speed of the logistic curve at $t_0$ and $\frac{1}{1+g}$ is the value of the logistic curve at $t_0$. To get the real value of the outcome $y_{i,j}$, the latent disease age $\psi_i(t)$ is first applied, then the longitudinal process from the latent disease age $\gamma_0(t)$ and finally a Gaussian noise $\varepsilon_{i,j}$ is added. We assume here that all the noises of visits are independent. The whole longitudinal process can thus be written as:

$$y_{i,j} = \gamma_0(\psi_i(t_{i,j})) + \varepsilon_{i,j} = \gamma_i(t_{i,j}) + \varepsilon_{i,j} \tag{3}$$

Note that using a Beta distribution for the noise instead of a Gaussian distribution would be more suited to a logistic trajectory. Nevertheless, in the context of small noise the Beta distribution could be approximated by a Gaussian distribution.

### 2.2.3 Modelling survival process

A Weibull distribution is used to model the survival probability from the latent disease age $\psi_i(t)$:

$$S_i(t) = S_0(\psi_i(t)) = \mathbb{1}_{\psi_i(t) > t_0} \exp\!\left(-\left(\frac{\psi_i(t) - t_0}{\nu}\right)^{\rho}\right) + \mathbb{1}_{\psi_i(t) \leq t_0}$$

where $\nu$ represents the variability of the distribution and $\rho$ the shape of the distribution. From there we also compute the individual hazard, which is, assuming that a patient has survived for a time $t$, the probability that he will not survive for an additional time $dt$:

$$h_i(t) = -\frac{S'_i(t)}{S_i(t)} = \mathbb{1}_{\psi_i(t) > t_0}\ \frac{\rho\, e^{\xi_i}}{\nu} \left(\frac{\psi_i(t) - t_0}{\nu}\right)^{\rho - 1}$$

### 2.2.4 Joint Temporal model

The proposed joint model referred to as the Joint Temporal model, is thus the combination of both a longitudinal sub-model $\gamma_i(t)$ and a survival sub-model $S_i(t)$ using the latent disease age $\psi_i(t)$ as a linkage structure, summarised in Table 1.

## 2.3 Estimation

### 2.3.1 Parameters

For estimation purpose, latent parameters $(z)$ are defined in addition to model parameters $(\theta)$ and hyperparameters $(\Pi)$. They can be summarised as follows for each patient $i$:

- **Latent parameters $(z)$:**
  - Latent fixed effects $(z_{fe})$: fixed effects sampled

$$\tilde{g} = \log(g) \sim \mathcal{N}\!\left(\bar{\tilde{g}},\ \sigma^2_{\tilde{g}}\right) \qquad \tilde{v}_0 = \log(v_0) \sim \mathcal{N}\!\left(\bar{\tilde{v}}_0,\ \sigma^2_{\tilde{v}_0}\right)$$

$$\tilde{\nu} = -\log(\nu) \sim \mathcal{N}\!\left(\bar{\tilde{\nu}},\ \sigma^2_{\tilde{\nu}}\right) \qquad \tilde{\rho} = \log(\rho) \sim \mathcal{N}\!\left(\bar{\tilde{\rho}},\ \sigma^2_{\tilde{\rho}}\right)$$

  - Latent random effects $(z_{re})$: random effects sampled

$$\xi_i \sim \mathcal{N}\!\left(\bar{\xi},\ \sigma^2_{\xi}\right) \qquad \tau_i \sim \mathcal{N}\!\left(\bar{\tau},\ \sigma^2_{\tau}\right)$$

- **Model parameters $(\theta)$:** fixed effects estimated from log-likelihood maximisation $\theta = \{\sigma_{\xi}, \sigma_{\tau}, t_0, \bar{\tilde{g}}, \bar{\tilde{v}}_0, \bar{\tilde{\nu}}, \bar{\tilde{\rho}}, \sigma\}$

- **Hyperparameters $(\Pi)$:** set by the user $\Pi = \{\sigma_{\tilde{g}}, \sigma_{\tilde{v}_0}, \sigma_{\tilde{\nu}}, \sigma_{\tilde{\rho}}\}$

To ensure identifiability, we set $\bar{\xi} = 0$ and $t_0 = \bar{\tau}$.

### 2.3.2 Log-likelihood

The likelihood estimated by the model is the following:

$$p(y, T_e, B_e \mid \theta, \Pi) = \int_z p(y, T_e, B_e, z \mid \theta, \Pi)\, dz$$

$p(y, T_e, B_e, z \mid \theta, \Pi)$ can be divided into two different terms: data attachment which represents how well the model describes the data $(y, t_e, B_e)$ and a prior attachment, which prevents over-fitting.

$$\log p((y, t_e, B_e), z, \mid \theta, \Pi) = \log p(y, t_e, B_e \mid z, \theta, \Pi) + \log p(z \mid \theta, \Pi)$$

The first term, data attachment, can be divided again into two terms considering that survival and longitudinal processes are independent regarding random effects. This is a quite common assumption in other papers ([Rizopoulos, 2012, Proust-Lima et al., 2017]). We can also separate the prior attachment term: two terms for the prior attachment of latent parameters (fixed and random) and one term for the prior attachment of model parameters. We end up with the following expression:

$$\log p((y, t_e, B_e), z, \theta \mid \Pi) = \log p(y \mid z, \theta, \Pi) + \log p(t_e, B_e \mid z, \theta, \Pi)$$
$$+ \log p(z_{re} \mid \theta, \Pi) + \log p(z_{fe} \mid \theta, \Pi)$$

The different log-likelihood parts with their different assumptions and the total formula of the log-likelihood is available in appendix A.

### 2.3.3 Algorithm

The first step is the estimation on the training dataset, it enables us to estimate fixed and associated random effects from a training data set. Directly maximising the log-likelihood has no analytical solution. Thus we use an Expectation-Maximization algorithm. Nevertheless, the computation of the expectation is also intractable due to the nonlinearity of the model. Thus, we use a Monte-Carlo Markov Chain Stochastic Approximation Expectation-Maximization (MCMC-SAEM) algorithm, as for the existing Longitudinal model. Its convergence has been proven by [Kuhn and Lavielle, 2004] for models that lie in the curved exponential family. The Joint Temporal model falls into such a category and further details are given in appendix B. To get the mean of the distribution of the model, we apply a Robbins-Monro convergence algorithm to the last iterations ([Robbins and Monro, 1951]). More details are given by [Koval, 2020] (p.41-43) and [Schiratti, 2017] (p.106). Latent parameters (defined in 2.3.1), are estimated during the estimating phase of the EM algorithm and model parameters during the maximisation phase, using sufficient statistics. The total log-likelihood, the sufficient statistics and the maximisation update rules, necessary for the computation, are given in the appendices A, B and C.

The second step is the validation on a test set, to compute the random effects for new patients. During this step, the prediction of random effects for the patients is estimated using the standard approach by maximising the posterior distribution of the random effects given the visits and the censored event. The solver *minimise* from the package Scipy ([Virtanen et al., 2020]) was used to maximise the log-likelihood. Note that for predictions, the survival probability is then corrected using the survival probability at the last visit as in other packages ([Rizopoulos, 2016]).

An implementation of the Joint Temporal model is available in the open-source library leaspy (v2): https://gitlab.com/icm-institute/aramislab/leaspy.

## 2.4 Reference models

### 2.4.1 Reference models

We chose to benchmark the Joint Temporal model against several reference models summurised with their equations in Table 1. First, we use one-process-only models. For the survival model, we use a Weibull AFT model to describe the survival process, using the Lifelines package ([Davidson-Pilon, 2023]). This model will be referred to as the AFT model. For the longitudinal model, we use the existing Longitudinal model described in part 2.2 (Equation (3)) using the open-source leaspy library https://gitlab.com/icm-institute/aramislab/leaspy. This model will be referred to as the Longitudinal model. We expect the Joint Temporal model to be at least as good as these two models.

Second, we use a two-stage model: a survival model that uses the random effects of the Longitudinal model as covariates ([Murawska et al., 2012]). Even though this model is subject to an immortal bias, it enables us to compare our model to a better survival model than the AFT model. We use the Longitudinal model to extract random effects for each individual, and then use them as covariates in a Weibull AFT model, using the Lifelines package ([Davidson-Pilon, 2023]). This model will be referred to as the Two-stage model, and the Joint Temporal model is expected to be at least as good as it.

Third, we use a joint model with shared random effects, to evaluate if the newly proposed structure could improve estimation. To do so, we use a logistic longitudinal process, using the JMbayes2 package ([Rizopoulos, 2016]). This model will be referred to as the JMbayes2 model.

# 3 Simulation: model validation

## 3.1 Method

For the simulation study we followed the ADEMP recommendation ([Morris et al., 2019]) and got inspiration from the work of ([Lavalley-Morelle et al., 2024]) also using MCMC-SAEM estimator.

### 3.1.1 Aims

The simulation study aimed to validate the Joint Temporal model. For model parameters $(\theta)$, we evaluate the estimation of the model parameters associated with their standard error. For random effects, we assessed the correlation between the estimated values and the simulated ones.

### 3.1.2 Data-generating mechanism

Data were simulated under the Joint Temporal model structure with the following procedure:

1. We simulated random effects using $\xi_i \sim \mathcal{N}\!\left(0, \sigma^2_{\xi}\right)$ and $\tau_i \sim \mathcal{N}\!\left(t_0, \sigma^2_{\tau}\right)$.

2. We modelled the age at first visit $t_{i,0}$ as $t_{i,0} = \tau_i + \delta_{f_i}$ with $\delta_{f_i} \sim \mathcal{N}\!\left(\bar{\delta}_f, \sigma^2_{\delta_f}\right)$.

3. We set a time of follow-up per patient $T_{f_i}$, with $T_{f_i} \sim \mathcal{N}\!\left(\overline{T_f}, \sigma^2_{T_f}\right)$ and a time between two visits $\delta_{v_{i,j}} \sim \mathcal{N}\!\left(\bar{\delta}_v, \sigma^2_{\delta_v}\right)$ to simulate $n_i$ visits until $t_{n_i} \leq t_{i,0} + T_{f_i} < t_{n_i+1}$.

4. We set the value of the outcome at each visit using a beta distribution of concentration $p$ and mode $\gamma_0(\psi_i(t_{i,j}))$ so that $y_{i,j} \sim \mathcal{B}(\gamma_0(\psi_i(t_{i,j})), p)$.

5. For each patient, we simulated the event $T_{e_i}$ through a Weibull distribution using $T_{e_i} \sim e^{-\xi_i} \mathcal{W}(\nu, \rho) + \tau_i$.

6. We considered that the event stopped the follow-up and that the follow-up censored the event. Thus all the visits after the event were censored: $t_{i,j} > T_{e_i}$ and events after the last visit were censored: $t_{i,\max(j)} < T_{e_i}$.

We simulated ALS real-like data using an ALS dataset, PRO-ACT, described in part 4.1.1, to get real-like values for parameters. Note that some parameters values were adjusted, such as the population estimated reference time, to make sure that no patient was left censored (Table 4 in appendix). Parameters directly associated with the disease have been extracted from data analysis, using the Longitudinal and AFT models (Figures 4, 5 in appendix). We simulated $M=100$ datasets with $N=200$ patients. The parameters used for the simulation study are summarised in Table 4 in appendix.