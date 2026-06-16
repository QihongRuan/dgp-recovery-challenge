# DGP Reverse-Engineering Report

**Dataset:** `data.csv`, n = 4000 observations, predictors `x1..x6`, continuous outcome `y`.
**Method:** OLS / robust IRLS / random-forest probing for structure discovery, then a joint maximum-likelihood fit of a heteroskedastic Student-t model. All numbers below are estimated from the data alone.

---

## 1. Functional form of E[y | x]

The conditional mean is fully described by **four terms** (plus intercept):

```
E[y | x] = 1.50  +  2.00·x1  −  0.81·x1²  +  1.50·sin(3·x2)  +  2.53·(x3·x4)
```

| Term            | Estimate | t-stat  | Notes |
|-----------------|----------|---------|-------|
| intercept       |  1.50    |  ~34    | |
| `x1`            |  2.00    | ~117    | strong linear |
| `x1²`           | −0.81    | −74     | concave (downward parabola) in x1 |
| `sin(3·x2)`     |  1.50    |  ~36    | **periodic**, amplitude ≈1.5, angular frequency = **3.0** (clean interior optimum on a frequency scan from 0.5 to 6; `cos(3·x2)` coefficient ≈ 0, so it is a pure sine, no phase shift) |
| `x3·x4`         |  2.53    |  ~54    | **interaction / gating** (see below) |

Mean-model fit: **R² = 0.858** (vs 0.55 for a naive linear model). A cross-validated random forest tops out at R² ≈ 0.83, i.e. there is no further smooth structure left to capture — the parametric form above is the signal.

### The `x3`–`x4` interaction (a latent regime gate)
`x3` has **no marginal effect on its own**. Estimating the `x3` slope separately within each `x4` group:

- `x4 = 0`:  x3 slope = **−0.05** (insignificant, t ≈ −1)  → x3 is inert
- `x4 = 1`:  x3 slope = **+2.52** (t ≈ 42)                 → x3 is strongly active

So `x4` acts as an on/off switch (a binary latent regime) that **turns on** the linear effect of `x3`. The product term `2.53·x3·x4` captures this exactly; adding a standalone `x3` term yields a coefficient of ≈0 (confirming no `x4=0` effect). `x4` itself has **no main effect** (coef ≈ 0.04, t ≈ 0.7).

### Why y is left-skewed
The marginal skew of y (−0.67) is *not* from skewed noise; it is induced by the concave `−0.81·x1²` term (which pushes the mean down for large |x1|) plus the one-sided activation of `x3·x4`. A simulation from the fitted DGP reproduces the data's skew (−0.72 vs −0.67), std (4.9 vs 5.0) and corr(y,x1)=0.72, corr(y,x3)=0.23.

---

## 2. Irrelevant predictors (zero effect on the mean)

- **`x5`** — zero effect. Linear coef insignificant; adds nothing to mean (and nothing to the noise scale, slope 0.0025). **Pure noise variable.**
- **`x6`** — zero effect. Insignificant everywhere. **Pure noise variable.**
- **`x2` (linear)** — `x2` enters the mean *only* through `sin(3·x2)`; its raw linear term is insignificant (t ≈ −0.2).
- **`x4` (main effect)** — no main effect; `x4` only matters by gating `x3`.
- **`x1·x4`** showed a marginal coefficient (≈ −0.11, |t| ≈ 3) but it is tiny, unstable under the heavy-tailed noise, and not retained — treated as a non-effect / artifact of the t-tails.

Net: only **x1, x2, x3, x4** drive the mean (x4 only via interaction). **x5 and x6 are inert.**

---

## 3. Error / noise structure

The noise is **heavy-tailed (Student-t)** and **heteroskedastic in x1**.

### Distribution: Student-t, df ≈ 2.4 — NOT Gaussian
Residuals have **excess kurtosis ≈ 29** and normality is decisively rejected (Shapiro W=0.74, Jarque–Bera ≈ 1.4×10⁵, KS p < 1e-60). Distribution comparison by AIC on the residuals:

| Distribution            | AIC      |
|-------------------------|----------|
| **Student-t (df≈2.0–2.4)** | **best (≈12,970)** |
| 2-component Gaussian mixture | 13,815 |
| Cauchy                  | 13,942   |
| Laplace                 | 13,976   |
| Gaussian                | 16,463 (far worst) |

A 2-component Gaussian mixture (≈88% with σ≈0.94, ≈12% with σ≈4.8) also fits reasonably and is an equivalent way to *describe* the contamination — about **12% of points behave like heavy-tail outliers** — but the single Student-t is the more parsimonious generative story.

### Variance is heteroskedastic — scales linearly with |x1|
Robust (MAD) noise scale across |x1| quartiles climbs monotonically: `0.49 → 0.84 → 1.22 → 1.49`. Modeling this:

```
σ(x) = 0.288 + 0.376·|x1|
```

(Equivalent clean forms fit almost identically: `σ ≈ 0.335·(1 + |x1|)`.) The variance does **not** depend on x2, x3, x4, x5, or x6 (robust scale is flat across all of them, e.g. σ by x4 = 0.93/0.93).

### Joint maximum-likelihood fit (final, self-contained)
Fitting `y = μ(x) + σ(x)·ε`, `ε ~ Student-t(df)` jointly by MLE:

```
μ(x) = 1.50 + 2.00·x1 − 0.81·x1² + 1.50·sin(3·x2) + 2.53·x3·x4
σ(x) = 0.288 + 0.376·|x1|
ε    ~ Student-t,  df = 2.43,  scale absorbed into σ
```

**Goodness of fit:** standardizing the residuals by σ(x) and comparing to a t(df=2.43), the KS test gives **p = 0.31 (cannot reject)** and the empirical vs theoretical quantiles match across the whole range (e.g. 1% quantile −5.73 vs −5.52; 99% quantile +5.80 vs +5.52). The model is fully adequate — no residual structure remains.

---

## 4. Other structure

**Predictor distributions (recovered):**
- `x1 ~ Uniform(−3, 3)`
- `x2 ~ Normal(0, 1)`
- `x3 ~ Normal(0, 1)`, **correlated with x2**: ρ ≈ 0.70, i.e. `x3 = 0.70·x2 + N(0, 0.72²)` (x2, x3 are jointly bivariate normal; all other predictors are independent of them).
- `x4 ~ Bernoulli(0.40)` — binary latent regime (gates x3).
- `x5 ~ Uniform(0, 5)` — inert.
- `x6 ~ Normal(0, 1)` — inert.

**Correlations among predictors:** only the `(x2, x3)` pair is correlated (0.70). All other pairwise correlations are ≈ 0.

**Outliers / contamination:** present and intrinsic — they are the heavy Student-t tails (equivalently ~12% "wide-component" points), not data-entry errors. Their rate is the same in both x4 regimes (≈3.8% of |z|>4 in each), i.e. contamination is homogeneous across the regime.

**Latent regimes / mixtures:** yes — `x4` is the latent binary regime variable that switches the `x3` effect on/off. There is no other hidden mixture in the mean.

---

## Final DGP specification (concrete)

```
# Predictors
x1 ~ Uniform(-3, 3)
x2 ~ Normal(0, 1)
x3 = 0.70*x2 + Normal(0, 0.72^2)      # corr(x2,x3) ≈ 0.70
x4 ~ Bernoulli(0.40)
x5 ~ Uniform(0, 5)                    # irrelevant
x6 ~ Normal(0, 1)                     # irrelevant

# Outcome
mu    = 1.50 + 2.00*x1 - 0.81*x1**2 + 1.50*sin(3*x2) + 2.53*x3*x4
sigma = 0.288 + 0.376*abs(x1)
y     = mu + sigma * StudentT(df = 2.43)
```

R²(mean) ≈ 0.86; noise: heavy-tailed Student-t (df≈2.4), heteroskedastic with SD growing linearly in |x1|; x5 and x6 are pure noise; x4 is a binary regime that gates x3.
