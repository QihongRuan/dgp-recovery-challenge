# Can an AI agent recover a hidden DGP from data alone?

A blind experiment: design a complex data-generating process (DGP), generate a
dataset, hand **only the data** to three frontier coding agents — **Claude**,
**Codex (GPT)**, and **Gemini** — and grade how much of the true generative
process each one can reverse-engineer.

## The idea behind it — in my own words

> simulation is a key method to validate the code. I learned this from Prof.
> Shuangge Ma — a Yale professor — in his high-dimensional statistics course at
> Xiamen University. how do you know your code is correct? someone may say,
> because I wrote it, line by line, it must be correct. their answer becomes
> because i write it so it must be correct. but the answer is simulations.
> Simulate a DGP, and see if the code can correctly estimate the DGP.
>
> — *Qihong Ruan*

This repo turns that lesson into a benchmark. Instead of asking "is *my* code
correct?", it asks: **given data from a known DGP, can an autonomous agent
recover the DGP?** Because we hold the ground truth, every claim an agent makes
is objectively gradeable — the same discipline Prof. Ma teaches for validating
estimators, applied to validating the *agents themselves*.

## How the experiment works

1. **`generator/generate.py`** builds a deliberately rich DGP (8 distinct,
   recoverable features) and writes `data/data.csv` (columns `x1..x6, y`).
   The ground truth lives in **`generator/TRUTH.md`** — the answer key.
2. Each agent is given an **identical, neutral prompt**
   (**`prompts/agent_prompt.txt`**) and *only* `data.csv`. No hints about any
   feature. Each runs in its own isolated directory and may write/run its own
   code.
3. Each agent writes a self-contained spec of the DGP it inferred — see
   **`reports/{claude,codex,gemini}_report.md`**.
4. Reports are scored against the 8-feature rubric — see **`results.md`**.

## The hidden DGP (the answer key)

```
y = 1.5 + 2.0·x1 − 0.8·x1²        (quadratic, concave in x1)
        + 1.5·sin(3·x2)           (sinusoid, frequency 3)
        + 2.5·x3·x4               (interaction — x3 acts only when x4 = 1)
        + 0·x5 + 0·x6             (two irrelevant decoy predictors)
    eps = (0.4 + 0.6·|x1|) · standardized_t(df=4)   (heteroskedastic, heavy-tailed)
        + [3% of rows] N(0, 8²)                      (outlier contamination)
    x3 = 0.7·x2 + noise           (collinearity, corr ≈ 0.70)
```

Eight gradeable features: a quadratic, a sinusoid, a conditional interaction,
two null predictors, collinearity, heteroskedasticity, heavy tails, and
contamination.

## Results (scored against `TRUTH.md`)

| Rank | Agent | Score | Verdict |
|---|---|---|---|
| 🥇 | **Codex** (GPT) | **7.8 / 8** | Recovered everything, incl. the exact 3% / σ=8 contamination as its own component |
| 🥈 | **Claude** | **7.4 / 8** | Equally perfect mean + noise form; folded the contamination into a single fat tail |
| 🥉 | **Gemini** | **6.1 / 8** | Two real misses: called the noise homoskedastic, and mistook the sinusoid for a step function |

Headline findings:

- **All three recovered the mean function to within ~1% of the true
  coefficients.** On the *signal*, the agents' inference is excellent.
- They diverged on the **noise**. The true error is `t(df=4)` *plus* a separate
  3% `N(0,8²)` shock. Claude and Codex absorbed both into a single heavier
  `t` (df≈2.4). Gemini split the noise into two mechanisms (a `t`-ish core + a
  5% `N(0,8²)` tail) and so recovered **df≈4.2 — closest to the true 4**, even
  though it finished last overall.
- The takeaway is exactly Prof. Ma's: **simulation reveals what inspection
  hides.** None of these agents could be "trusted because they wrote the code
  line by line" — feeding them a *known* DGP exposed precisely where each one's
  inference is strong (mean structure: all flawless) and where it is fragile
  (noise decomposition: genuinely model-dependent).

Full per-feature scoring and analysis in **[`results.md`](results.md)**.

## Reproduce

```bash
python3 generator/generate.py        # regenerates data/data.csv (seed fixed)
# then give data/data.csv + prompts/agent_prompt.txt to any agent, blind
```

Requires `numpy`, `pandas`, `scipy`. The seed (`20260616`) is fixed, so the
dataset is byte-for-byte reproducible.

## Layout

```
dgp-recovery-challenge/
├── README.md                 # this file
├── results.md                # scored head-to-head + analysis
├── data/
│   └── data.csv              # the blind dataset (what each agent sees)
├── generator/
│   ├── generate.py           # the DGP (the "answer key" code)
│   └── TRUTH.md              # ground-truth spec + grading rubric
├── prompts/
│   └── agent_prompt.txt      # identical neutral prompt given to every agent
└── reports/
    ├── claude_report.md
    ├── codex_report.md
    └── gemini_report.md
```

---

*Inspired by Prof. Shuangge Ma's high-dimensional statistics course at Xiamen
University (Prof. Ma is on the faculty at Yale).*
