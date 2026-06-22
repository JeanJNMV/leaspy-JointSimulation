# About the dataset (part of PRO-ACT)

The paper does not use the full PRO-ACT database. The authors applied several successive filtering criteria that substantially reduced the sample size. Here is the filtering pipeline:

1. Out of the 8,571 patients from the PRO-ACT database, they subselected 6,034 patients with sex and first symptoms (spinal or bulbar onset) provided.

2. Out of them, 2,219 had their first visit with a Mitos score equal to 0. This was done deliberately: To limit left-censored VNI initiation, they selected patients with a Mitos score equal to 0.

3. Then 42 patients were dropped for the Analysis dataset due to left censored VNI. For the Benchmark dataset, they also dropped patients with less than 3 visits and ended up with 1,919 patients.

So the final datasets used were:
- **Analysis dataset**: 2,177 patients (for the application study and parameter estimation for simulation)
- **Benchmark dataset**: 1,919 patients (for the model comparison, requiring ≥3 visits)

In summary, the main reason for the drastic reduction from ~8,500 to ~2,200 patients is the **Mitos score = 0 requirement at baseline**, which ensured patients were at an early disease stage to limit left-censored NIV initiation events. Additional filters for available covariates (sex, onset site) and minimum visit counts further reduced the sample.