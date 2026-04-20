# Hydra Backbone Experiments — Plan (v2)

Revision notes over v1: the "multi-stage" framing was conflating three different mechanisms. This version separates them cleanly, adds a deep-supervision ablation for MS-TCN, and adds a cross-dataset section with capacity-matched configurations.

All experiments train under the **offsets-only** Hydra setup (classification head for per-frame active/inactive + regression head for start/end offsets, GIoU loss).

---

## 1. Conceptual clarification: three different "stages"

Reading the literature (and my own v1 plan), the word "stages" is overloaded and it's worth being explicit about what each mechanism does. Conflating them invites bad comparisons.

| Mechanism | Where | What it does | What it is NOT |
|---|---|---|---|
| **Iterative refinement** | MS-TCN `n_stages` | Stage $N$ takes the *predictions* of stage $N-1$ (after softmax) and smooths them. Operates at full temporal resolution throughout. | Not multi-scale; not hierarchical. |
| **Deep supervision** | MS-TCN loss `multi_layer` | Loss is computed on every refinement stage, not just the last one. Forces all stages to produce valid predictions. | Orthogonal to `n_stages` — you can have one without the other. |
| **Hierarchical downsampling** | MS-Transformer `n_branch_layers` | Each branch layer halves the temporal resolution, producing a feature pyramid. Operates on features, not predictions. | Not iterative refinement; the stages do not try to refine each other's output. |

**Consequence:** you cannot compare "MS-TCN with 4 stages" against "MS-Transformer with 5 branch layers" as if they were two flavors of the same thing. The former is a refinement loop in prediction space; the latter is a multi-resolution encoder in feature space. My v1 plan put them in the same RQ table — that was wrong.

---

## 2. Research questions (revised)

| # | Question | Answered by |
|---|----------|-------------|
| RQ1 | Does MS-TCN 4×10 replicate on LSFB? | Group A |
| RQ2a | How much does MS-TCN's iterative refinement loop contribute? | Group A |
| RQ2b | Is MS-TCN's deep-supervision loss (`multi_layer: true`) important? | Group A |
| RQ3 | How do RNN, Transformer, and hierarchical Transformer backbones compare to MS-TCN with reasonable defaults? | Group B |
| RQ4 | What are good hyperparameters for the Transformer family (depth, width, attention pattern, positional encoding)? | Group C |
| RQ5 | Does multi-resolution temporal processing help Transformers (ViT vs. MS-Transformer)? | Group D |
| RQ6 | Do architecture rankings transfer across sign languages and dataset regimes (LSFB → DGS, Phoenix, BOBSL)? | Group E |

RQ2a and RQ5 are now separate on purpose. RQ5 is the "flat vs. hierarchical attention" question — the transformer-native analogue of a pyramid. It has nothing to do with iterative refinement.

---

## 3. Experiment groups

### Group A — MS-TCN: replication + refinement + deep supervision (5 runs)

| Config | `n_stages` | deep sup. (`multi_layer`) | Purpose |
|---|---|---|---|
| `mstcn_4s10l.yaml` | 4 | false | **Chapter baseline replication** |
| `mstcn_1s10l.yaml` | 1 | false (irrelevant, single stage) | No refinement |
| `mstcn_2s10l.yaml` | 2 | false | 1 refinement step |
| `mstcn_3s10l.yaml` | 3 | false | 2 refinement steps |
| `mstcn_4s10l_deepsup.yaml` | 4 | **true** | Refinement + deep supervision (closer to original MS-TCN paper) |

The last row is new. Your existing config has `multi_layer: false` which means loss is only computed on the final stage. The original MS-TCN paper's main contribution was actually the deep supervision, so it's worth testing whether that changes the multi-stage story on Hydra. If deep supervision helps, it should also be enabled for the MS-Transformer (see Group D).

### Group B — Cross-family head-to-head at reasonable defaults (4 new runs + 1 from A)

One config per family, all offsets-only, all on LSFB.

| Config | Family | Summary |
|---|---|---|
| `mstcn_4s10l.yaml` (from A) | TCN | 4 stages × 10 layers × 64 hidden |
| `rnn_gru.yaml` | RNN | Bi-GRU, 4 layers, 128 hidden |
| `rnn_lstm.yaml` | RNN | Bi-LSTM, 4 layers, 128 hidden |
| `vit_base.yaml` | Transformer | 4h × 6L × 256, window-128, RoPE |
| `mstx_base.yaml` | MS-Transformer | 2 stem + 5 branch, 256 hidden, 4 heads |

Note: in v1 I suggested skipping one of the RNN variants. Keeping both because (a) GRU vs. LSTM is a cheap test and (b) the earlier chapter already reports both, so symmetry is useful.

### Group C — Transformer hyperparameter sweep (4 new runs + 1 from B)

All inherit from `vit_base.yaml`; each varies one axis at a time.

| Config | Axis | Change |
|---|---|---|
| `vit_base.yaml` (from B) | capacity | 4h × 6L × 256 (baseline) |
| `vit_small.yaml` | capacity | 4h × 4L × 128 |
| `vit_large.yaml` | capacity | 8h × 8L × 512 |
| `vit_full_attn.yaml` | attention | no sliding window (full) |
| `vit_sinusoidal.yaml` | positional encoding | sinusoidal instead of RoPE |

### Group D — Flat vs. hierarchical Transformer (1 new run + 1 from B)

This is NOT a refinement ablation. It's a question about whether multi-resolution features help Transformers on dense temporal labeling.

| Config | Summary |
|---|---|
| `vit_base.yaml` (from B) | Flat attention, full resolution throughout |
| `mstx_base.yaml` (from B) | 2 stem + 5 branch, temporal downsampling ×2 per branch |
| `mstx_large.yaml` | 3 stem + 6 branch, 512 hidden, 8 heads |

If the winner is MS-Transformer and deep supervision helped in Group A, consider adding `multi_layer: true` to the MS-Transformer's best config too — its output is also a list per stage, so deep supervision is applicable.

---

## 4. LSFB run inventory

**13 unique LSFB runs.**

```
Group A (5):   mstcn_4s10l, mstcn_1s10l, mstcn_2s10l, mstcn_3s10l, mstcn_4s10l_deepsup
Group B (4):   rnn_gru, rnn_lstm, vit_base, mstx_base  (+ mstcn_4s10l)
Group C (4):   vit_small, vit_large, vit_full_attn, vit_sinusoidal  (+ vit_base)
Group D (1):   mstx_large  (+ vit_base, mstx_base)
```

---

## 5. Group E — Cross-dataset generalization

### 5.1 The capacity-data mismatch problem

Running the LSFB winner unchanged on Phoenix would be an unfair test. Phoenix has:

- ~7-second videos (vs. LSFB's multi-minute conversations)
- ~1,081 gloss types in a restricted weather-report domain (vs. LSFB's open vocabulary)
- Automatically aligned gloss boundaries derived from statistical transcript alignment (vs. LSFB's manual annotation)

A 5M-parameter Transformer that wins on 40 hours of LSFB will almost certainly overfit on Phoenix. Ignoring this and reporting "Transformer underperforms on Phoenix" as a cross-language finding conflates the language effect with the data-size effect.

**Proposed approach:** blend two framings.

1. **Capacity-matched for the main comparison.** On Phoenix, report a *small* variant of the winning family alongside the MS-TCN baseline. The claim is "architecture ranking is preserved *when capacity is matched to dataset size*", which is a stronger and more honest empirical statement than raw transfer.
2. **Document the overfitting gap.** Also run the *full-capacity* winner on Phoenix and report the difference. This turns the overfitting from an embarrassment into a finding: "Phoenix exposes when Transformer capacity outruns the data; the gap between big-ViT and small-ViT is X points of mF1s."

### 5.2 Dataset regime table

| Dataset | Size | Annotation | Capacity regime | Finalist configs |
|---|---|---|---|---|
| **LSFB** | ~40h | manual, naturalistic | full | all 13 (architecture search) |
| **DGS Corpus** | ~50h | manual, strict | full | 3 finalists (best of each family) |
| **Phoenix WT** | ~9h, short clips | auto-aligned | **small** | 3 small finalists + 1 full-capacity for overfitting diagnostic |
| **BOBSL subset** | varies | approximated boundaries | depends | 3 finalists; caveat on boundary-precision metrics |

### 5.3 Cross-dataset run inventory

Assuming the three finalists from LSFB are: MS-TCN best (likely `mstcn_4s10l` or `mstcn_4s10l_deepsup`), Transformer best (likely `vit_base` or `vit_large`), hierarchical Transformer best (likely `mstx_base` or `mstx_large`).

- **DGS (3 runs):** all three full-capacity finalists retrained.
- **Phoenix (4 runs):** small versions of the three finalists + one full-capacity diagnostic (e.g., `vit_base_phoenix` alongside `vit_small_phoenix`).
- **BOBSL (3 runs):** all three full-capacity finalists retrained. Note: report mF1b more prominently than mF1s because the boundaries are approximated.

**Cross-dataset total: 10 runs.**

### 5.4 BOBSL boundary-quality caveat

BOBSL boundaries come from forced alignment or heuristic estimation rather than manual annotation. Any metric that requires precise boundary placement (mF1s at 70% IoU, mIoU above ~80%) is measuring two things at once: model error plus annotation noise. When reporting on BOBSL:

- Lead with mF1b (boundary-distance-based), which tolerates a few-frame offset.
- Report mF1s at the lower IoU thresholds (40–50%) rather than the full 40–75% range.
- Treat any gap between BOBSL and the manually-annotated corpora as partly an annotation-noise artefact, not just a generalization failure.

### 5.5 Config inheritance pattern for cross-dataset runs

Each cross-dataset override is a thin file that inherits from an existing LSFB architecture config and replaces only the `datasets` block, the `experiment.id`, and optionally the variant name. See the three example files in this bundle:

- `vit_base_dgs.yaml` — straightforward "same arch, different dataset" case.
- `vit_small_phoenix.yaml` — shows the capacity-matched approach for small-data regimes.
- `mstcn_4s10l_bobsl.yaml` — shows MS-TCN on BOBSL with a comment about the boundary-quality caveat.

If you end up running every finalist on every dataset, you'll have `3 archs × 3 datasets = 9` such files plus the Phoenix diagnostic, which is what the "10 runs" figure corresponds to. If your framework supports a CLI-level dataset override you could avoid those files entirely, but I've written them as YAMLs to match your existing pattern.

---

## 6. Total inventory

| Group | Runs | Dataset |
|---|---|---|
| A | 5 | LSFB |
| B | 4 new | LSFB |
| C | 4 new | LSFB |
| D | 1 new | LSFB |
| E | 10 | DGS / Phoenix / BOBSL |
| **Total** | **24** | — |

Plus 3× reruns on the top finalists with different seeds for statistical significance = **~27 runs total** for the complete chapter extension. That's large but tractable on a single A100 given each run is modest-sized (most are < 10M params) and windowed.

---

## 7. Execution order

1. **Group A first.** Validates pipeline, answers the refinement + deep-supervision questions. Picks the best MS-TCN config.
2. **Group B.** Cross-family comparison with the best MS-TCN as TCN entry.
3. **Group C** on the winning Transformer family.
4. **Group D** resolves flat vs. hierarchical.
5. Select three LSFB finalists.
6. **Group E** runs the three finalists × DGS, Phoenix (+ small variants + diagnostic), BOBSL.
7. Pick top-3 across all experiments and rerun with 3 seeds for error bars.

---

## 8. Open questions to resolve before launching

Unchanged from v1:

1. **Head `in_features` vs. backbone output width.** Your existing `transformer01.yaml` uses a 2:1 ratio (backbone=256, head in=128). I'm using 1:1 in all base configs. Verify one way or the other and fix the four base configs if needed — variants will inherit.
2. **`max_length: 300` vs. `window_size: 3500`.** Possible mismatch; verify frame rate and any subsampling.
3. **`multi_layer` loss default.** Your existing code has it off. Group A's `mstcn_4s10l_deepsup.yaml` tests whether this should be the default. If it wins, revisit the Group B MS-Transformer baseline.
4. **Phoenix windowing.** Videos are short (~7s); the LSFB window of 3500 frames is almost certainly the full video. The `data_phoenix.yaml` template uses a smaller window — sanity-check the number.
5. **BOBSL subset selection.** "Approximated boundaries" means different things depending on which subset and which pseudo-label pipeline. Document which subset is used and how boundaries were obtained, because this directly affects what mF1s means on that dataset.
