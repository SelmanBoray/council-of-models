# Measuring the judges

## Motivation

When Andrej Karpathy released [llm-council](https://github.com/karpathy/llm-council), he noted that the member models fairly consistently picked GPT-5.1 as the best answer and Claude as the worst, while his own reading of the same answers was different: GPT-5.1 struck him as too wordy, Claude as too terse. In other words, the council had a consistent taste, and it was not his.

That is the general problem with LLM-as-judge setups. A ranking produced by models is easy to get and easy to over-trust. This project keeps every ranking and computes a few numbers that tell you how much to trust them. This page defines those numbers exactly, as implemented in [`src/council/analytics.py`](../src/council/analytics.py), and lists what they cannot tell you.

## What the data looks like

In each run, every surviving member reviews all surviving answers, its own included, under anonymous labels (`Response A`, `Response B`, ...), and returns a complete order from best to worst. The labels are mapped back to model ids before storage, so every stored review is a list of model ids, best first.

Notation used below:

- `k` is the number of answers ranked in a run.
- `pos_r(j, m)` is the 1-based position that judge `j` gave model `m`'s answer in run `r` (1 = best, `k` = worst).
- A review is **usable** if its ranking is non-empty. Failed and unparseable reviews are ignored everywhere. If a judge somehow has two usable reviews in one run, only the first counts.

All results are rounded to 4 decimals.

## Council ranking (per run)

The council's aggregate order for one run sorts models by their mean position over all usable reviews in that run, self-votes included:

```text
score_r(m) = mean over judges j of pos_r(j, m)
```

The lowest score is the council's #1. Ties are broken alphabetically by model id; members that nobody ranked come last. A run with no usable review has no council #1.

## Leaderboard

For each model `m`, over all stored runs that have at least one usable review:

```text
mean_rank(m)   = mean of pos_r(j, m) over every usable review (r, j) that ranked m
appearances(m) = number of such reviews
wins(m)        = number of runs in which m is the council's #1
```

Rows are sorted by `mean_rank` (best first), ties by model id. Note that `appearances` counts reviews, not runs: one run of a four-member council contributes four appearances per model. Self-votes are included, so a self-favoring model gets a slightly better `mean_rank` than its peers alone would give it; the self-preference index below tells you how much.

## Self-preference index

For model `m` in run `r`, let `P_r(m)` be the set of *other* judges with a usable review that ranked `m`. Run `r` counts for `m` only if `m` produced a usable review that contains itself and `P_r(m)` is not empty. Over those runs `R_m`:

```text
SP_r(m) = mean over j in P_r(m) of pos_r(j, m)  -  pos_r(m, m)

SP(m)   = mean over r in R_m of SP_r(m)
```

- **Positive**: `m` puts its own answer higher than its peers do (it favors itself).
- **Zero**: `m` places itself where its peers place it.
- **Negative**: `m` is harder on itself than its peers are.

In a council of `k` answers, `SP_r(m)` lies between `-(k-1)` and `+(k-1)`. Results are sorted from most to least self-favoring.

### Why self-inclusion is deliberate

As in the original llm-council, every judge ranks every answer, its own among them. This project keeps that on purpose, tells the judge so in the review prompt ("one of them may be your own... do not try to guess which response is yours"), and builds its main metric on it:

1. **It is what makes self-preference measurable.** If a judge never ranks its own answer, you only ever see the peer side of the comparison, and `SP` cannot be computed.
2. **Every ballot covers the same answers.** Positions mean the same thing for every judge, and Kendall tau (below) compares rankings of identical sets.
3. **It measures the bias that matters in practice.** The judge does not see model names, so `SP` captures self-preference under anonymity: recognizing and preferring its own style, reasoning or phrasing. Favoritism by name is excluded by construction.

The cost is that self-votes count toward the council ranking and the leaderboard, one vote out of `k`. A strongly self-favoring judge therefore distorts the aggregate, and `SP` is the number that tells you by how much.

### One biased judge leaks into everyone's score

`SP` is relative to the other judges, so one judge's bias shows up in other models' scores too. The offline `fake` provider demonstrates this on purpose: `fake:alpha` always ranks itself first, while `fake:beta` and `fake:gamma` rank by a shared notion of quality and always put alpha last. With `council ask ... --fake`, any number of runs gives:

| model        | SP    | why |
| ------------ | ----: | --- |
| `fake:alpha` | +2.00 | ranks itself 1st; both peers rank it 3rd |
| `fake:beta`  | +0.50 | honest about itself, but alpha's ballot pushes it down one place |
| `fake:gamma` | +0.50 | same as beta |

Beta and gamma never promote themselves. In alpha's ballot, though, both are pushed down one place because alpha jumped over them, so their peer mean gets worse by `1/(k-1) = 0.5` while their self-rank stays honest. In general, read `SP` across the whole council: a single large outlier with everyone else slightly positive points at the outlier.

## Reviewer agreement matrix

For two judges `a` and `b` with usable reviews in the same run `r`, take the `n` models both rankings contain (`n >= 2`). Count the pairs of models both judges order the same way (concordant, `C`) and the pairs they order oppositely (discordant, `D`):

```text
tau_r(a, b) = (C - D) / (n (n - 1) / 2)

tau(a, b)   = mean over shared runs r of tau_r(a, b)
```

Every review is a strict order without ties, so this equals Kendall's tau-b. `+1` means identical order, `0` no relation, `-1` reversed order. The matrix is symmetric with `1.0` on the diagonal; a pair of judges that never reviewed the same run has no entry.

With the fake council, `tau(beta, gamma) = 1.00` (they share one quality order) and `tau(alpha, beta) = tau(alpha, gamma) = -0.33`: the two rankings agree on beta versus gamma and disagree on both pairs involving alpha, so `(1 - 2) / 3`.

## Human vs council

When you vote, you pick the best answer before names are revealed (the web UI enforces this; `council vote` and the MCP `council_vote` tool simply record a pick). Let `V` be the stored runs that have a human vote and a council #1 (at least one usable review):

```text
n                  = |V|
agreement_rate     = |{ r in V : human_vote_r = council_top_r }| / n      (none if n = 0)
per_model_human_wins(m) = |{ r in V : human_vote_r = m }|
```

Runs where you revealed the names without voting have no vote and do not count. This is the number Karpathy's observation is about: if it stays low across many runs, the council's preferences are not yours, however well the judges agree with each other.

## Caveats

These metrics measure the judges against each other and against you. None of them measures correctness.

- **Small samples.** There are no confidence intervals yet. With `k = 3`, `SP_r` moves in steps of 0.5, so a few runs are mostly noise. Treat anything under a few dozen runs on varied questions as anecdotal.
- **Position bias.** The label order is shuffled per run (seeded by the run id), but within one run every judge sees the same order. Position effects average out only across many runs.
- **Verbosity and style bias.** The review prompt tells judges not to reward length, style or position. That does not stop them. If every judge prefers long answers, they agree with each other (high `tau`), nobody shows self-preference, and the council's taste still drifts away from yours. Shared biases are invisible to `SP` and `tau`; only the human-vs-council rate can reveal them.
- **Shared training data.** Models trained on overlapping data make correlated judgments. High agreement is not independent confirmation.
- **Self-recognition.** Anonymization hides names, not writing style. A model may recognize its own answer, which is part of what `SP` measures, but it means "blind" review is only partly blind.
- **Debate blurs authorship.** After debate rounds, answers have absorbed each other's ideas, so "its own answer" is less well defined and `SP` is harder to interpret.
- **Parser fill-in.** If a reviewer leaves labels out of its ranking, the missing ones are appended in label order. A reviewer that often does this produces partly artificial rankings; check the stored `raw` text.
- **Mixed council sizes.** Positions are not normalized. `SP` ranges and mean ranks from councils of different sizes are on different scales, and stats aggregate every run in `data_dir`. Use a separate `data_dir` per council configuration.
- **Selection effects.** Failed answers and failed reviews are excluded. A model that fails more often on hard questions is judged on an easier subset.
- **The human is one human.** Your vote is a preference, not ground truth, and reading the reviews first can anchor it. The web UI hides the council's verdict and the final synthesis until you vote, and warns you on the Reviews tab.
