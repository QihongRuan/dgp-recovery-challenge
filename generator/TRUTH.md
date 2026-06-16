# GROUND TRUTH — secret. Agents never see this.

n = 4000, seed = 20260616

## Predictors
- x1 ~ Uniform(-3, 3)
- x2 ~ Normal(0, 1)
- x3 = 0.7*x2 + sqrt(1-0.49)*N(0,1)   (corr ~0.70)
- x4 ~ Bernoulli(0.4)
- x5 ~ Uniform(0, 5)
- x6 ~ Normal(0, 1)

## Outcome
y = 1.5 + 2.0*x1 - 0.8*x1^2 + 1.5*sin(3*x2) + 2.5*x4*x3 + eps
eps = (0.4 + 0.6*|x1|) * standardized_t(df=4) + [3% chance] N(0, 8^2)

## Grading checklist (1 point each, partial allowed)
| ID | Feature | What counts as "found it" |
|----|---------|---------------------------|
| F1 | Quadratic concave effect of x1 | identifies x1 AND x1^2, negative curvature |
| F2 | Sinusoidal effect of x2 (freq~3) | identifies nonlinear/periodic in x2 (freq partial credit) |
| F3 | Interaction x4*x3 | x3 matters only when x4=1 (or names x3:x4 interaction) |
| F4 | x5, x6 irrelevant | excludes both / coeffs ~0 |
| F5 | Collinearity corr(x2,x3)~0.7 | notes x2,x3 correlated |
| F6 | Heteroskedasticity sd ~ |x1| | notes non-constant variance growing with |x1| |
| F7 | Heavy-tailed errors (t, df~4) | notes non-Gaussian / fat-tailed residuals |
| F8 | 3% outlier contamination | notes outliers / mixture / contamination |

Score = features found / 8.  Bonus: correct coefficient magnitudes.
