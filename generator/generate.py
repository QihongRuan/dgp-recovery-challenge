#!/usr/bin/env python3
"""
SECRET data-generating process for the DGP-recovery challenge.
The agents (Claude / Codex / Gemini) NEVER see this file.
They only get share/data.csv (columns x1..x6, y) and must reverse-engineer
everything below from the data alone.

Ground-truth feature checklist (used for grading):
  F1  x1 has a QUADRATIC (concave) effect:        +2.0*x1 - 0.8*x1^2
  F2  x2 enters through a SINUSOID (freq ~3):      +1.5*sin(3*x2)
  F3  INTERACTION x4*x3: x3 only matters when x4=1: +2.5*x4*x3
  F4  x5 and x6 are IRRELEVANT (zero coefficient)
  F5  COLLINEARITY: corr(x2, x3) ~ 0.7
  F6  HETEROSKEDASTICITY: sd grows with |x1|:       sigma = 0.4 + 0.6*|x1|
  F7  HEAVY-TAILED errors: standardized Student-t, df=4
  F8  OUTLIER CONTAMINATION: 3% of rows get +N(0, 8^2) shock
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20260616)
n = 4000

# --- predictors -----------------------------------------------------------
x1 = rng.uniform(-3, 3, n)
x2 = rng.normal(0, 1, n)
# F5 collinearity: x3 correlated ~0.7 with x2
x3 = 0.7 * x2 + np.sqrt(1 - 0.7**2) * rng.normal(0, 1, n)
x4 = rng.binomial(1, 0.4, n)          # binary group
x5 = rng.uniform(0, 5, n)             # irrelevant
x6 = rng.normal(0, 1, n)             # irrelevant

# --- mean function --------------------------------------------------------
mu = (1.5
      + 2.0 * x1
      - 0.8 * x1**2                   # F1 quadratic
      + 1.5 * np.sin(3.0 * x2)        # F2 sinusoid
      + 2.5 * x4 * x3)                # F3 interaction (x3 matters only if x4=1)
#   + 0*x5 + 0*x6                     # F4 irrelevant

# --- noise ----------------------------------------------------------------
sigma = 0.4 + 0.6 * np.abs(x1)        # F6 heteroskedastic
t_raw = rng.standard_t(df=4, size=n)
t_std = t_raw / np.sqrt(4 / (4 - 2))  # F7 standardize t(4) to unit variance
eps = sigma * t_std

# F8 outlier contamination
mask = rng.random(n) < 0.03
eps = eps + mask * rng.normal(0, 8, n)

y = mu + eps

df = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, y=y))
df = df.round(5)
# write next to the repo's data/ dir regardless of where the script is run from
import os
out = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
df.to_csv(out, index=False)
print("wrote", os.path.normpath(out), df.shape)
print(df.describe().round(3).T[["mean", "std", "min", "max"]])
print("corr(x2,x3) =", round(np.corrcoef(x2, x3)[0, 1], 3))
print("outlier rows =", int(mask.sum()))
