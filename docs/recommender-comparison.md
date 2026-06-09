# Recommender Comparison: LLM vs. Heuristic

## Thesis

The `ClaudeRecommender` earns its place not by outperforming `BasicRecommender`
on clean-signal clusters — it should not, and the harness confirms it does not.
Its value is threefold:

1. **Drafting.** When a profile is new or poorly calibrated, a human can ask the
   LLM to propose thresholds from first principles rather than tuning them by
   hand.
2. **Adjudicating conflicting evidence.** On ambiguous clusters — moderate
   positive signal alongside non-trivial contamination — a fixed threshold
   is forced to pick a side. The LLM can *describe the conflict*, recommend
   subclustering, and explain which evidence pulls in which direction. That
   narrative is auditable in a way a single threshold number is not.
3. **Explaining decisions.** Every `Recommendation` carries a `reasoning`
   field. The heuristic populates it with a template string. The real LLM
   fills it with evidence-specific prose that a reviewer can read and dispute.

On clean-signal clusters the two recommenders agree. That is the *expected*
result, not a limitation.

---

## Methodology

### The harness

`scripts/recommender_comparison.py` runs both recommenders on the same
`pd.DataFrame` (the consensus table produced by `score_evidence`) and returns
a per-cluster DataFrame with columns:

| Column | Description |
|---|---|
| `cluster` | cluster ID |
| `heuristic` | `BasicRecommender` decision |
| `llm` | `ClaudeRecommender` decision |
| `agree` | `True` when both match |
| `llm_reasoning` | prose from the LLM |
| `heuristic_reasoning` | template string from the heuristic |

The `compare(table, profile, llm_client=...)` function accepts an injectable
client, which makes the harness testable without an API key.

### The consensus table

`build_demo_table()` constructs four synthetic clusters designed to exercise
each case the harness is meant to illustrate:

| Cluster | pass_pan_t_frac | is_contaminant_frac | Intended category |
|---------|-----------------|---------------------|-------------------|
| A | 0.82 | 0.02 | Clean keep — strong signal, low contamination |
| B | 0.05 | 0.55 | Clean drop — no signal, high contamination |
| C | 0.35 | 0.22 | Ambiguous — both recommenders purify, via different thresholds |
| D | 0.30 | 0.45 | Designed divergence — heuristic drops (contam > 0.4), mock LLM purifies (contam < 0.5) |

Cluster D is the key demonstration: the heuristic's 0.4 cutoff and the LLM's
notional 0.5 boundary produce *different* decisions on the same evidence. A
real LLM would explain *why* the evidence is ambiguous rather than silently
picking a side.

### Mock vs. real mode

The harness ships with `FakeClaudeClient`, a deterministic mock that:
- parses the table text embedded in the user message by `ClaudeRecommender._build_user_message`;
- applies slightly different thresholds than `BasicRecommender` so the harness
  has a real disagreement to surface on cluster D;
- returns the exact `{"content": [{"type": "text", "text": "```json...```"}]}`
  shape that `ClaudeRecommender.recommend` expects.

**To run the real comparison** (requires `pip install 'rarecell[agent]'` and
`ANTHROPIC_API_KEY` set in the environment):

```python
import os
from anthropic import Anthropic
from scripts.recommender_comparison import compare, build_demo_table, _make_profile

# The real Anthropic client exposes messages.create, not messages_create.
# Wrap it to match the protocol ClaudeRecommender expects.
class AnthropicShim:
    def __init__(self, client):
        self._client = client
    def messages_create(self, messages, **kw):
        resp = self._client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=messages,
            **kw,
        )
        return {"content": [{"type": b.type, "text": b.text} for b in resp.content]}

client = AnthropicShim(Anthropic())
table = build_demo_table()
profile = _make_profile()
result = compare(table, profile, llm_client=client)
print(result.to_string())
```

### How to run the harness

```bash
# Mock mode (no API key — safe in CI):
uv run python scripts/recommender_comparison.py

# Tests:
uv run pytest tests/test_recommender_comparison.py -q
```

---

## Results

> **Note:** The numbers below are PLACEHOLDERS. Replace them after running the
> harness on real study data and measuring downstream precision/recall.

### Decision agreement on synthetic demo table (mock mode)

| Cluster | Heuristic | LLM (mock) | Agree | Notes |
|---------|-----------|------------|-------|-------|
| A | keep | keep | yes | Clean keep — expected agreement |
| B | drop | drop | yes | Clean drop — expected agreement |
| C | purify | purify | yes | Ambiguous — both recommend subclustering |
| D | drop | purify | **NO** | Designed divergence (see thesis §2) |

**Mock-mode agreement rate: 3/4 (75%)**
The single disagreement is the designed one (cluster D). On real data the
real-LLM agreement rate on clean clusters is expected to be > 95%; the
interesting signal lives in the residual disagreements on mixed clusters.

### Downstream precision / recall (PLACEHOLDER — real data required)

| Recommender | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| BasicRecommender | — | — | — | Measure on held-out labeled data |
| ClaudeRecommender (real) | — | — | — | Same held-out data, no mock |

Fill these in after running `compare()` with the real Claude client on a
labeled benchmark (e.g., PBMC3k with ground-truth T-cell cluster annotations).

---

## What this shows / doesn't

**What it shows:**
- The heuristic and LLM agree on high-confidence clusters. Disagreements are
  concentrated in the ambiguous middle, exactly where a fixed threshold is
  least reliable.
- The LLM's `reasoning` field provides auditable, cluster-specific prose. The
  heuristic's reasoning is a template. For a reviewer deciding whether to
  accept a `purify` recommendation, the LLM's explanation is qualitatively
  more useful.
- The harness is runnable in CI without an API key, so the comparison
  structure can be validated continuously even before a real LLM run.

**What it doesn't show:**
- Whether the real LLM's decisions are *more accurate* than the heuristic's
  on labeled data. That requires downstream measurement (fill in the
  precision/recall table above).
- Whether the LLM's reasoning is *correct* — prose can be confident and wrong.
  The reasoning field is a starting point for human review, not a replacement
  for it.
- Performance at scale. The real `ClaudeRecommender` makes one API call per
  `recommend()` invocation. On large studies with many clusters, batching or
  caching may be necessary.
