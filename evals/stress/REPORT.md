# Real-provider stress experiment report

## Executive summary

The recorded OpenAI experiment contains **909 attempted turns**, of which **680 committed** and **229 were rejected** with `invalid_estimate` / `estimate_validation`. Among the 680 committed turns with latency, estimator P50 was **7,254.95 ms** and nearest-rank P95 was **21,121.28 ms**. The sum of the 680 recorded cost values is **$3.6485556 USD**; it is an estimated recorded total, not an invoice or the total cost of all attempts. Rejected turns have unknown costs and may have incurred charges.

All 680 committed cache observations say `none`: exact and semantic hit rates are each **0/680 = 0%** of committed turns with known cache kind. The 229 rejected rows have no cache observation and are excluded, not assumed misses. The mean memory-recall proxy is **0.956621** across **525 scored committed turns**, equally weighted by turn rather than by fact. It is not a direct measure of whether a model answered a later question correctly.

## Summary by scenario and attachment size

One row per scenario × nominal attachment size. Latency percentiles and cache rates use committed turns only; total cost sums recorded values and omits unknown costs of rejected attempts. Recall is the mean nonblank `memory_score`, with the number of scored turns in parentheses. Both hit rates are 0 because the exact and semantic cache backends exist in `app/cache/` but are not wired into the estimation path, so every committed turn reports `cache_hit_kind=none`.

| Scenario | Attachment (KiB) | Attempted | Committed | P50 latency (ms) | P95 latency (ms) | Total recorded cost (USD) | Exact hit rate | Semantic hit rate | Mean recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| growing | 0 | 76 | 56 | 6,674.19 | 12,467.44 | 0.032071 | 0/56 | 0/56 | 0.901 (44) |
| growing | 5 | 76 | 64 | 7,209.44 | 14,992.19 | 0.110989 | 0/64 | 0/64 | 1.000 (52) |
| growing | 20 | 76 | 65 | 7,579.29 | 20,682.29 | 0.325789 | 0/65 | 0/65 | 1.000 (53) |
| growing | 50 | 76 | 54 | 9,867.13 | 31,740.28 | 0.573461 | 0/54 | 0/54 | 0.951 (42) |
| growing | 100 | 76 | 57 | 11,499.43 | 30,090.19 | 0.665328 | 0/57 | 0/57 | 1.000 (45) |
| pivot | 0 | 76 | 60 | 5,749.21 | 11,130.58 | 0.034776 | 0/60 | 0/60 | 0.935 (48) |
| pivot | 5 | 76 | 55 | 6,080.15 | 13,215.24 | 0.089138 | 0/55 | 0/55 | 0.971 (43) |
| pivot | 20 | 76 | 58 | 8,051.60 | 27,215.60 | 0.283178 | 0/58 | 0/58 | 0.980 (46) |
| pivot | 50 | 76 | 57 | 7,898.21 | 24,747.81 | 0.598704 | 0/57 | 0/57 | 0.961 (45) |
| pivot | 100 | 76 | 56 | 10,673.27 | 21,142.49 | 0.591205 | 0/56 | 0/56 | 0.980 (44) |
| contradiction | 0 | 36 | 24 | 6,204.18 | 7,787.77 | 0.011486 | 0/24 | 0/24 | 0.729 (16) |
| contradiction | 5 | 36 | 23 | 5,874.34 | 8,534.19 | 0.026624 | 0/23 | 0/23 | 0.922 (15) |
| contradiction | 20 | 36 | 22 | 6,428.45 | 8,616.06 | 0.075560 | 0/22 | 0/22 | 0.905 (14) |
| contradiction | 50 | 29 | 21 | 7,495.75 | 12,190.90 | 0.177954 | 0/21 | 0/21 | 0.885 (14) |
| contradiction | 100 | 12 | 8 | 9,954.42 | 13,715.20 | 0.052293 | 0/8 | 0/8 | 1.000 (4) |

Recorded cost by scenario: growing **$1.707637**, pivot **$1.597002**, contradiction **$0.343917** (fewer attempted rows).

## Coverage and measurement

Source: [`evals/stress/results.csv`](results.csv), SHA-256 `cc049a16ef667f670b223af5344d4e41df9f14c91121622a981ff7ca38cb286f`. Every row is labeled `real-http`; the CSV does not identify the actual model ID or version. The local synthetic HTTP service invokes the estimator, not the public HTTP endpoint. Its latency is the monotonic elapsed time of the estimator await, including locking, memory preparation, provider calls and commit; it excludes attachment extraction, scoring, CSV writing and HTTP transport. Latency is available only for successful commits, so the distribution has success-selection bias and says nothing about failed-turn latency.

The dataset has growing 380, pivot 380 and contradiction 149 attempted rows; scenario lengths N=3, 6 and 10 are represented, but **N=1 and N=20 are missing**. Nominal attachment sizes 0, 5, 20, 50 and 100 KiB and repeats 1–4 appear, but coverage is incomplete and uneven: there are 155 distinct `(scenario, length, attachment size, repeat)` runs and not every run reaches its configured length. Extracted attachment character counts are respectively 0, 5,875, 23,503, 58,759 and 58,759; nominal size, PDF bytes and extracted characters are different measurements. These data do not establish why the last two extracted counts match. The requested exercise matrix is therefore not fully covered.

`scenario_turn` is the attempted position within a scenario; `turn_index` counts committed session turns and can lag after a rejection. Rejection rows count as attempts, retain extraction counts, and have blank commit-derived observations, tokens, cost, latency and memory score. Blank is unknown, never a known zero. The fixed rejection reason does not identify the particular validation rule or cause. Unrecorded interrupted work, if any, is also outside this CSV.

The memory proxy uses the implementation in [`evals/stress/metrics.py`](metrics.py) and [`evals/stress/run.py`](run.py): for each earlier successfully committed scenario fact, it checks whether its literal phrase occurs case-insensitively in a scalar value of the current committed snapshot's summary, anchors or metadata. A row's `memory_score` is the fraction found among those eligible earlier facts. It is blank without eligible facts or on rejection. Conflicting facts are tracked as asserted, not resolved as current truth. The table averages nonblank row scores equally; it is not a pooled fact-level recall or a semantic recall test. See also [`stress-runner.md`](../../docs/components/stress-runner.md).

The three stress metrics (`LatencyBudgetMetric`, `CostBudgetMetric`, `MemoryDriftMetric`) live in `evals/stress/metrics.py` rather than `evals/metrics.py`: they evaluate a `turn_observed` observation or a session snapshot, not an `EstimationResult`, so they share no inputs with the golden-dataset metrics. They reuse the same `MetricResult` shape (`name`, `score`, `passed`, `details`) and stay deterministic: case-insensitive literal matching, no embeddings and no LLM judge.

## Latency versus tokens

Total tokens are `tokens_in + tokens_out` per committed row. Bins are left-inclusive and right-exclusive; each row has both token counts and latency. P50 is the conventional median; P95 is sorted value at one-based rank `ceil(0.95 × n)`.

| Total tokens | Committed samples | P50 latency (ms) | P95 latency (ms) |
|---|---:|---:|---:|
| 0–<5,000 | 166 | 6,267.50 | 9,661.28 |
| 5,000–<10,000 | 62 | 6,521.08 | 12,467.44 |
| 10,000–<20,000 | 86 | 6,965.81 | 15,495.68 |
| ≥20,000 | 366 | 9,078.51 | 24,317.60 |

## Recorded cumulative cost versus scenario turn

Each point is the mean, over runs with a row at that scenario turn, of the sum of **recorded** cost values in that run through that position. A blank is omitted from the recorded sum, not imputed as a known zero; therefore these are partial recorded-cost sums, not actual total costs. The final column counts runs at each point with at least one unknown cost through that position. The changing sample counts reflect unequal configured and observed run lengths: points are **not matched cohorts**. They should not be summed across turns.

| Scenario turn | Runs observed | Mean cumulative recorded USD/run | Runs with unknown cost to date |
|---:|---:|---:|---:|
| 1 | 155 | 0.00173035 | 52 |
| 2 | 155 | 0.00410667 | 90 |
| 3 | 155 | 0.00704753 | 106 |
| 4 | 95 | 0.01065137 | 71 |
| 5 | 95 | 0.01553412 | 75 |
| 6 | 94 | 0.02070070 | 74 |
| 7 | 40 | 0.02755485 | 32 |
| 8 | 40 | 0.03492177 | 32 |
| 9 | 40 | 0.04313250 | 32 |
| 10 | 40 | 0.05185932 | 32 |

## Memory-recall proxy versus configured N

N is the configured scenario length, not the current attempted or committed turn. The denominator is the number of nonblank per-turn scores in each N group; unscored attempts are excluded.

| Configured N | Scored turns | Mean memory proxy |
|---:|---:|---:|
| 1 | 0 | Not covered |
| 3 | 68 | 0.992647 |
| 6 | 187 | 0.931818 |
| 10 | 270 | 0.964727 |
| 20 | 0 | Not covered |

## Where my CAG starts to break and why

The CAG breaks first on **latency and cost, not on memory**. Because every turn re-sends history, summary, anchors and the extracted attachment, mean `tokens_in` per committed turn grows from **15,850 at scenario turn 1 to 65,408 at turn 10 (4.1×)**, and mean cost per committed turn grows from **$0.002604 to $0.010267 (3.9×)**. Turns 7–10 come only from N=10 runs, so the cohorts differ, but the trend is monotonic at every step. The attachment dominates from the first turn: with a 100 KiB PDF, turn 1 already sends a mean 32,333 input tokens versus 1,288 without one (25×) and costs 12× more. Latency follows: median latency is 9,078.51 ms for turns of ≥20,000 tokens versus 6,267.50 ms below 5,000, and P95 exceeds 20 s in every growing and pivot cell with an attachment of 20 KiB or more. Against a `LatencyBudgetMetric(budget_ms=4000)` SLA, **0 of 680 committed turns pass**, so with this provider the latency contract is broken from turn 1 regardless of size. Separately, **25% of attempts (229/909) failed hard** with an invalid structured estimate; that rate does not grow with attachment size (22.9–27.1%) or turn (15–34%), so it is a schema-adherence problem of the provider, not a context-length effect.

Memory degrades only mildly and silently. The fact-tracker proxy stays at 1.000 at turn 2 and dips to **0.911–0.913 at turns 5–6** (0.947 at turn 4), right after the pivot scenario changes the stack at turn 5 and the contradiction scenario has stated its first budget at turn 3; most misses come from those two scenarios (43 of 57 imperfect rows). The contradiction scenario without attachment is the weakest cell (0.729 mean recall). No run reached N=20, so this experiment cannot say where `project_name` is lost in long conversations. The limit case that justifies RAG is therefore **large or multiple attachments plus long conversations**: by turn 10 the prompt already carries ~65k tokens and ~4× the per-turn cost of turn 1, all of it re-sent every turn, while only a small part of the attachment is relevant to each question. Retrieving the relevant chunks instead of injecting the full extracted text attacks the dominant cost and latency term directly; it would not, by itself, fix the 25% invalid-estimate rate.

## Reproduce the aggregates

From the repository root, the following standard-library-only command reads the CSV and prints the source hash, coverage, summary and the three table inputs. It makes no provider calls or file changes. Displayed report values are rounded from these outputs.

```sh
.venv/bin/python - <<'PY'
import collections, csv, hashlib, math, statistics
from pathlib import Path
p = Path('evals/stress/results.csv')
rows = list(csv.DictReader(p.open(newline='', encoding='utf-8')))
print('sha256', hashlib.sha256(p.read_bytes()).hexdigest(), 'rows', len(rows))
for key in ('mode', 'scenario', 'length', 'attachment_size_kib', 'repeat',
            'status', 'reason', 'cache_hit_kind'):
    print(key, dict(collections.Counter(r[key] for r in rows)))
def vals(key, rs=rows):
    return [float(r[key]) for r in rs if r[key] != '']
def p95(v):
    return sorted(v)[math.ceil(.95 * len(v)) - 1]
for key in ('latency_ms', 'cost_usd', 'memory_score'):
    v = vals(key)
    print(key, 'n', len(v), 'sum', sum(v), 'mean', statistics.mean(v),
          'median', statistics.median(v), 'p95', p95(v))
for low, high in ((0, 5000), (5000, 10000), (10000, 20000), (20000, math.inf)):
    v = [float(r['latency_ms']) for r in rows
         if r['latency_ms'] and r['tokens_in'] and r['tokens_out']
         and low <= int(r['tokens_in']) + int(r['tokens_out']) < high]
    print('tokens', low, high, 'n', len(v), 'p50', statistics.median(v), 'p95', p95(v))
for n in (1, 3, 6, 10, 20):
    v = vals('memory_score', [r for r in rows if int(r['length']) == n])
    print('recall N', n, 'n', len(v), 'mean', statistics.mean(v) if v else None)
runs = collections.defaultdict(list)
for r in rows:
    runs[tuple(r[k] for k in ('scenario', 'length', 'attachment_size_kib', 'repeat'))].append(r)
print('runs', len(runs))
for t in sorted({int(r['scenario_turn']) for r in rows}):
    cohort = [g for g in runs.values() if any(int(r['scenario_turn']) == t for r in g)]
    prefixes = [[r for r in g if int(r['scenario_turn']) <= t] for g in cohort]
    totals = [sum(vals('cost_usd', prefix)) for prefix in prefixes]
    unknown = sum(any(not r['cost_usd'] for r in prefix) for prefix in prefixes)
    print('turn', t, 'runs', len(cohort), 'mean recorded', statistics.mean(totals), 'unknown', unknown)
for size in sorted({int(r['attachment_size_kib']) for r in rows}):
    print('attachment', size, 'chars', sorted({int(r['attachments_total_chars'])
          for r in rows if int(r['attachment_size_kib']) == size}))
PY
```
