# Results — head-to-head scoring

Each agent received only `data/data.csv` and the identical neutral prompt in
`prompts/agent_prompt.txt`. Reports were graded against the 8-feature rubric in
`generator/TRUTH.md`. Partial credit allowed.

## Leaderboard

| Rank | Agent | Score |
|---|---|---|
| 🥇 | **Codex** (GPT) | **7.8 / 8** |
| 🥈 | **Claude** | **7.4 / 8** |
| 🥉 | **Gemini** | **6.1 / 8** |

## The answer key

```
y = 1.5 + 2.0·x1 − 0.8·x1² + 1.5·sin(3·x2) + 2.5·x3·x4
    + (0.4 + 0.6·|x1|) · standardized_t(df=4)
    + [3% of rows] N(0, 8²) contamination
x3 = 0.7·x2 + noise  (corr 0.70);  x5, x6 irrelevant
n = 4000, seed = 20260616
```

## Per-feature scoring

| # | True feature | Claude | Codex | Gemini |
|---|---|:--:|:--:|:--:|
| F1 | quadratic `−0.8·x1²` | ✅ −0.81 | ✅ −0.81 | ✅ −0.8 |
| F2 | **sinusoid `1.5·sin(3·x2)`** | ✅ exact, freq=3 | ✅ exact, period 2π/3 | ❌ modeled as `sign(x2)`+linear |
| F3 | interaction `2.5·x3·x4` | ✅ 2.53 | ✅ 2.53 | ✅ 2.5 |
| F4 | x5, x6 irrelevant | ✅ | ✅ | ✅ |
| F5 | corr(x2,x3)=0.70 | ✅ 0.70 | ✅ 0.703 | ✅ 0.7 |
| F6 | **heteroskedastic `σ∝|x1|`** | ✅ `0.29+0.38|x1|` | ✅ `0.3+0.4|x1|` | ❌ "constant variance" |
| F7 | heavy-tailed t(df=4) | ✅ t, df≈2.4 | ✅ t, df≈2.5 | ✅ t, **df≈4.2** |
| F8 | 3% · N(0, 8²) outliers | ◐ "~12% fat tail" | ✅ "2.9%, sd 8.25" | ✅ "5%, N(0,8²)" |
| | **Total** | **7.4** | **7.8** | **6.1** |

## Analysis

**Mean structure — a clean sweep.** All three agents recovered the conditional
mean to within ~1% of the true coefficients (1.50, 2.00, −0.80, 1.50, 2.50),
*except* Gemini on the `x2` term. Claude and Codex both found the sinusoid by
scanning frequencies and locking onto freq = 3 with amplitude 1.5; both isolated
the `x3·x4` gate (x3 inert unless x4 = 1); both flagged x5 and x6 as pure noise.
On the signal, the agents' estimation machinery is excellent.

**The df / contamination trade-off — the most interesting result.** The true
error is `t(df=4)` *plus* a separate 3% `N(0,8²)` shock. These two ingredients
are nearly likelihood-equivalent to a single heavier-tailed `t`:

- **Claude** and **Codex** each fit one fat `t`, landing at **df≈2.4** (too
  heavy) because the single distribution has to soak up the extra outliers.
  Codex additionally surfaced a 3-component mixture whose smallest component was
  **2.9% with sd 8.25** — a near-exact hit on the hidden `3% · N(0,8²)` — but
  preferred the parsimonious single-`t` as its headline.
- **Gemini** modeled the noise as two mechanisms (a `t`-ish core + a **5%
  `N(0,8²)`** tail) and consequently recovered **df≈4.2**, the closest to the
  true 4. Its "thin-t + explicit contamination" story is arguably *truer* to the
  actual generative process — even though Gemini scored lowest overall.

The data underdetermines the model: "one fat t" and "thin t + contamination" are
two defensible decompositions of the same likelihood. Different agents picked
different — but reasonable — stories.

**Gemini's two genuine failures.** (1) It ran a heteroskedasticity check, found
nothing, and concluded constant variance — a false negative on a feature both
rivals nailed cleanly (`σ ∝ |x1|`). (2) It mistook `sin(3·x2)` for a step
function `sign(x2)` plus a linear term, missing the periodicity entirely. These
are real inference errors, not stylistic differences.

## Why this matters (Prof. Ma's point)

Simulation reveals what inspection hides. Because we held the ground truth, we
could see exactly where each agent's statistical reasoning is strong (mean
structure: all flawless) and where it is fragile (noise decomposition: genuinely
model-dependent; Gemini missed heteroskedasticity outright). No agent could be
"trusted because it wrote the code line by line" — but a known DGP makes every
claim falsifiable, and turns "do you trust it?" into "here is its score."
