"""WDR / WAR metrics from Min et al. (VAC, ICCV 2021) as individual
torchmetrics, designed for torchmetrics.MetricCollection.

Each metric is a scalar:
  WERAuxiliary         -> WER*_a  (visual-only classifier F_a)
  WERPrimary           -> WER*_p  (contextual classifier F_p)
  WordDeteriorationRate-> WDR     (correct in HYP_a, wrong in HYP_p)
  WordAmeliorationRate -> WAR     (wrong in HYP_a, correct in HYP_p)

Identity (Eq. 7 of the paper):  WER*_p = WER*_a + WDR - WAR.

All four metrics share the same `update()` (a three-way alignment of
reference / auxiliary hypothesis / primary hypothesis) and the same states,
so MetricCollection groups them and runs the alignment only once per batch.

update(refs, hyps_aux, hyps_primary): each argument is a list (one entry per
sample) of token sequences -- lists of gloss IDs (ints) or gloss strings,
whatever your decoder emits after CTC collapse.

Note (paper, footnote 1): the three-sentence alignment (WER*) can differ
slightly from plain pairwise WER, since merging the two alignments can split
a substitution into a deletion + insertion. Use torchmetrics.text.
WordErrorRate for the standard WER and these metrics for the diagnostic.
"""

from typing import Hashable, List, Optional, Sequence, Tuple

import torch
from torchmetrics import Metric

GAP = None  # gap symbol in alignments
Token = Hashable
Column = Tuple[Optional[Token], Optional[Token]]  # (ref, hyp)


# --------------------------------------------------------------------------
# Alignment helpers
# --------------------------------------------------------------------------

def _align(ref: Sequence[Token], hyp: Sequence[Token]) -> List[Column]:
    """Levenshtein alignment of hyp to ref; returns columns (ref_tok, hyp_tok),
    with None marking a gap (deletion/insertion)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    cols: List[Column] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            cols.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            cols.append((ref[i - 1], GAP)); i -= 1          # deletion
        else:
            cols.append((GAP, hyp[j - 1])); j -= 1          # insertion
    cols.reverse()
    return cols


def _merge_on_ref(
    align_a: List[Column], align_p: List[Column]
) -> List[Tuple[Optional[Token], Optional[Token], Optional[Token]]]:
    """Merge two pairwise alignments sharing the same reference into rows
    (ref*, hyp_a*, hyp_p*). Insertions from either side become columns where
    ref* is a gap."""
    merged = []
    ia, ip = 0, 0
    while ia < len(align_a) or ip < len(align_p):
        a_ins = ia < len(align_a) and align_a[ia][0] is GAP
        p_ins = ip < len(align_p) and align_p[ip][0] is GAP
        if a_ins:
            merged.append((GAP, align_a[ia][1], GAP)); ia += 1
        elif p_ins:
            merged.append((GAP, GAP, align_p[ip][1])); ip += 1
        else:
            ref_tok = align_a[ia][0]
            merged.append((ref_tok, align_a[ia][1], align_p[ip][1]))
            ia += 1; ip += 1
    return merged


# --------------------------------------------------------------------------
# Base metric: shared update / states
# --------------------------------------------------------------------------

class _PredictionInconsistencyBase(Metric):
    """Shared state and update for the VAC inconsistency metrics.

    Identical update + states across subclasses lets MetricCollection place
    them in a single compute group (alignment runs once per batch).
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("n_ref", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_a", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_wdr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_war", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        refs: List[Sequence[Token]],
        hyps_aux: List[Sequence[Token]],
        hyps_primary: List[Sequence[Token]],
    ) -> None:
        for ref, ha, hp in zip(refs, hyps_aux, hyps_primary):
            ref, ha, hp = list(ref), list(ha), list(hp)
            merged = _merge_on_ref(_align(ref, ha), _align(ref, hp))
            self.n_ref += len(ref)
            for r, a, p in merged:
                a_ok = a == r  # gap matching gap counts as correct
                p_ok = p == r
                self.err_a += not a_ok
                self.err_p += not p_ok
                self.n_wdr += a_ok and not p_ok
                self.n_war += p_ok and not a_ok

    def _ratio(self, numerator: torch.Tensor) -> torch.Tensor:
        return numerator / self.n_ref.clamp(min=1)


# --------------------------------------------------------------------------
# Public metrics (one scalar each)
# --------------------------------------------------------------------------

class WERAuxiliary(_PredictionInconsistencyBase):
    """WER*_a: word error rate of the auxiliary (visual-only) classifier."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.err_a)


class WERPrimary(_PredictionInconsistencyBase):
    """WER*_p: word error rate of the primary (contextual) classifier."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.err_p)


class WordDeteriorationRate(_PredictionInconsistencyBase):
    """WDR: fraction of glosses correct in HYP_a but wrong in HYP_p."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.n_wdr)


class WordAmeliorationRate(_PredictionInconsistencyBase):
    """WAR: fraction of glosses wrong in HYP_a but correct in HYP_p."""

    higher_is_better = True

    def compute(self) -> torch.Tensor:
        return self._ratio(self.n_war)


# --------------------------------------------------------------------------
# Example usage / smoke test (reproduces Fig. 4 of the paper)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from torchmetrics import MetricCollection

    metrics = MetricCollection(
        {
            "wer_aux": WERAuxiliary(),
            "wer_primary": WERPrimary(),
            "wdr": WordDeteriorationRate(),
            "war": WordAmeliorationRate(),
        },
        compute_groups=True,  # default; alignment runs once for all four
    )

    REF = "__ON__ HEUTE NACHT MEHR SCHNEE NORD SUEDOST ABER KALT".split()
    HYPa = "__ON__ HEUTE NACHT SCHNEE NORD SUEDOST ABER".split()
    HYPp = "__ON__ HEUTE NACHT MEHR SCHNEE NORD SUED SUEDOST SUED ABER KALT".split()

    metrics.update([REF], [HYPa], [HYPp])
    print({k: round(float(v), 4) for k, v in metrics.compute().items()})
    # expected: all four = 2/9 ~= 0.2222
    print("compute groups:", metrics.compute_groups)