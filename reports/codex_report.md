# Reverse-engineered DGP for `data.csv`

## Executive specification

The data are best described by

```text
y = 1.5 + 2.0*x1 - 0.8*x1^2 + 1.5*sin(3*x2) + 2.5*(x3*x4)
    + (0.3 + 0.4*|x1|)*eta

eta ~ Student-t(df about 2.5, location 0, scale 1)
```

The fitted, non-rounded heteroskedastic Student-t MLE was:

```text
E[y | x] =
  1.5008
  + 1.9972*x1
  - 0.8098*x1^2
  + 1.5007*sin(3*x2)
  + 2.5291*x3*x4

error scale s(x) = 0.2878 + 0.3757*|x1|
eta = error / s(x) ~ t_df=2.425
```

So the exact-looking rounded DGP is the first displayed formula. The sine term has period `2*pi/3` in `x2`. The only interaction in the conditional mean is `x3*x4`: `x3` matters only when the binary variable `x4` equals 1.

## Mean-function evidence

I fit broad and targeted feature libraries including raw predictors, powers, pairwise interactions, sine/cosine terms, and `x4`-gated nonlinearities.

The selected core OLS fit was:

| term | coefficient | SE | p-value |
|---|---:|---:|---:|
| intercept | 1.5337 | 0.0450 | 5.5e-224 |
| `x1` | 2.0123 | 0.0171 | <1e-300 |
| `x1^2` | -0.8110 | 0.0110 | <1e-300 |
| `sin(3*x2)` | 1.5185 | 0.0424 | 7.3e-244 |
| `x3*x4` | 2.5257 | 0.0468 | <1e-300 |

This model had `R^2 = 0.8577` and residual RMSE `1.8936`. A plain linear model had holdout RMSE about `3.47`; adding the quadratic/sinusoidal/interaction structure reduced that to about `1.88-1.89`.

Nested checks under the heteroskedastic Student-t likelihood supported the same core model:

| added terms beyond core | likelihood-ratio p | delta AIC | delta BIC | interpretation |
|---|---:|---:|---:|---|
| `x1*x4` | 0.319 | +1.0 | +7.3 | not real |
| all linear main effects `x2,x3,x4,x5,x6` | 0.967 | +9.1 | +40.5 | not real |
| `x4*x6`, `x2*x5`, `x1*x4` | 0.310 | +2.4 | +21.3 | not real |
| `x5` Fourier terms through frequency 4 | 0.384 | +7.5 | +57.8 | not real |
| extra `x2` Fourier terms | 0.013 | -4.7 | +14.1 | small overfit by AIC, rejected by BIC |
| `x1^3` | 0.018 | -3.6 | +2.7 | tiny overfit; coefficient about 0.011 |

The broad lasso screen selected the core terms first (`x3*x4`, `x1`, `sin(3*x2)`, `x1^2`) and then only very small unstable terms. BIC and the t-likelihood both favor the simple core formula.

## Irrelevant predictors

In the conditional mean:

- `x5` is irrelevant.
- `x6` is irrelevant.
- `x2` has no linear effect; it matters only through `sin(3*x2)`.
- `x3` and `x4` have no main effects; they matter through the interaction `x3*x4`.
- No threshold effects were detected, apart from the binary gating by `x4`.
- No meaningful interactions were detected other than `x3*x4`.

Tree permutation importances agreed: `x1`, `x2`, `x3`, and `x4` carried signal, while `x5` and `x6` were effectively zero. For example, an ExtraTrees model gave permutation importances approximately `x1=1.35`, `x2=0.05`, `x3=0.14`, `x4=0.11`, `x5=0.001`, `x6=0.000`.

## Error and noise structure

The errors are emphatically not Gaussian and are heteroskedastic.

Using the core mean model, residual diagnostics were:

```text
residual sd       1.8938
median abs dev    0.9325
skew              0.1267
excess kurtosis  29.00
1% / 99% quantiles: -4.99, 4.94
0.1% / 99.9% quantiles: -13.09, 16.09
```

Normality is rejected decisively:

```text
D'Agostino K^2 p = 3.9e-250
Jarque-Bera p    = 0
Anderson normal statistic = 197.3, far above 1% critical value 1.035
```

Distribution likelihoods for the core residuals:

| model | fitted parameters | AIC | BIC |
|---|---|---:|---:|
| Gaussian, constant scale | sd 1.894 | 16463 | 16476 |
| Laplace, constant scale | scale 1.055 | 13977 | 13989 |
| Student-t, constant scale | df 2.02, scale 0.775 | 13597 | 13616 |
| Student-t, `s(x)=a+b*x1^2` | df 2.36, `s=0.440+0.137*x1^2` | 13049 | 13100 |
| Student-t, `s(x)=a+b*|x1|` | df 2.43, `s=0.288+0.376*|x1|` | 12982 | 13033 |

Thus the error is best modeled as

```text
epsilon | x = (0.2878 + 0.3757*|x1|) * t_2.425
```

or, in rounded generating-process form,

```text
epsilon | x = (0.3 + 0.4*|x1|) * t_2.5.
```

For `df = 2.425`, the conditional variance exists but is very large:

```text
Var(epsilon | x) = [df/(df-2)] * s(x)^2
                 approximately 5.70 * (0.2878 + 0.3757*|x1|)^2.
```

The fourth moment does not exist for this fitted t distribution, explaining the extreme sample kurtosis and occasional large residuals.

After scaling by the rounded `0.3 + 0.4*|x1|`, the standardized residuals fit a t distribution well:

```text
t.fit(error / (0.3 + 0.4*|x1|)):
  df = 2.426, location = -0.026, scale = 0.948

fixed df comparison on standardized residuals:
  df=2.0 AIC 14487
  df=2.5 AIC 14466
  df=3.0 AIC 14489
  df=4.0 AIC 14582
```

Auxiliary regressions of standardized residual scale on the raw predictors found no remaining meaningful heteroskedasticity:

```text
z = residual / (0.3 + 0.4*|x1|)

regress z^2 on raw predictors:   R^2 = 0.00071, p = 0.828
regress |z| on raw predictors:   R^2 = 0.00101, p = 0.671
regress log(z^2) on predictors:  R^2 = 0.00063, p = 0.868
```

## Mixtures, outliers, and regimes

The raw residuals contain outliers, but they are explained by the heavy-tailed t error and the `|x1|` scale function. A homoskedastic two-normal mixture approximation was:

```text
88.1% N(-0.030, 0.941^2) + 11.9% N(0.221, 4.840^2)
```

A three-normal mixture split this further into roughly:

```text
76.3% sd 0.814, 20.8% sd 2.270, 2.9% sd 8.249
```

However, these homoskedastic mixtures had worse information criteria than the heteroskedastic t model. I found no evidence for separate mean regimes or predictor-driven contamination beyond the `x1`-dependent scale.

## Predictor structure

The predictors themselves appear to have this structure:

```text
x1 ~ Uniform(-3, 3)
x5 ~ Uniform(0, 5)
x4 ~ Bernoulli(0.4)
x2, x3 ~ approximately bivariate normal, each mean 0 and sd 1, corr about 0.703
x6 ~ N(0, 1)
all other pairwise correlations are near zero
```

Empirical checks:

- `x1` range was `[-2.999, 3.000]`; KS test against fixed `Uniform(-3,3)` gave p `0.542`.
- `x5` range was `[0.002, 4.999]`; KS test against fixed `Uniform(0,5)` gave p `0.369`.
- `x4` sample mean was `0.401`, consistent with Bernoulli `p=0.4`.
- Normality checks for `x2`, `x3`, and `x6` had p-values `0.308`, `0.446`, and `0.661`.
- The only substantial predictor correlation was `corr(x2,x3)=0.703`; all other predictor correlations were near zero.

## Final DGP

My final concrete reconstruction is:

```text
x1 ~ U(-3, 3)
x5 ~ U(0, 5)
x4 ~ Bernoulli(0.4)
(x2, x3) ~ bivariate normal with means 0, variances 1, corr 0.7
x6 ~ N(0, 1)

y | x = 1.5 + 2*x1 - 0.8*x1^2 + 1.5*sin(3*x2) + 2.5*x3*x4
        + (0.3 + 0.4*|x1|)*eta

eta ~ Student-t(df = 2.5, location = 0, scale = 1)
```

The mean-relevant predictors are `x1`, `x2`, `x3`, and `x4`, with `x2` only through a sinusoid and `x3,x4` only through their interaction. The mean-irrelevant predictors are `x5` and `x6`. The noise is heavy-tailed and heteroskedastic, with variance increasing approximately linearly in `|x1|` through the scale term.
