"""Edit-operation breakdown of WER as individual torchmetrics.

Provides the del/ins/sub decomposition that papers report next to WER
(e.g. "del/ins WER" columns on PHOENIX14) and that torchmetrics'
WordErrorRate does not expose:

  DeletionRate      -> #del / #ref
  InsertionRate     -> #ins / #ref
  SubstitutionRate  -> #sub / #ref
  WordErrorRateOps  -> (#del + #ins + #sub) / #ref   (= sum of the three)

All four share the same update (one Levenshtein alignment per sample) and
states, so MetricCollection groups them and aligns only once per batch.

update(refs, hyps): each argument is a list (one entry per sample) of token
sequences -- lists of gloss IDs or gloss strings.

Reading the breakdown (CTC "health"):
  * high deletions  -> blank-dominated spike outputs, signs being skipped;
  * high insertions -> over-segmentation / hallucinated glosses;
  * substitutions   -> visually confusable or rare glosses.
"""

from typing import Hashable, List, Sequence

import torch
from torchmetrics import Metric

from slp.metrics.continuous_recognition.vac import GAP, _align  # reuse the Levenshtein backtrace

Token = Hashable


class _EditOpsBase(Metric):
    """Shared state and update: counts edit operations against the reference."""

    full_state_update = False
    higher_is_better = False
    is_differentiable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("n_ref", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_del", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_ins", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_sub", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        refs: List[Sequence[Token]],
        hyps: List[Sequence[Token]],
    ) -> None:
        for ref, hyp in zip(refs, hyps):
            ref, hyp = list(ref), list(hyp)
            self.n_ref += len(ref)
            for r, h in _align(ref, hyp):
                if h is GAP:
                    self.n_del += 1
                elif r is GAP:
                    self.n_ins += 1
                elif r != h:
                    self.n_sub += 1

    def _ratio(self, numerator: torch.Tensor) -> torch.Tensor:
        return numerator / self.n_ref.clamp(min=1)


class DeletionRate(_EditOpsBase):
    """#deletions / #reference glosses."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.n_del)


class InsertionRate(_EditOpsBase):
    """#insertions / #reference glosses."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.n_ins)


class SubstitutionRate(_EditOpsBase):
    """#substitutions / #reference glosses."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.n_sub)


class WordErrorRateOps(_EditOpsBase):
    """Standard WER, computed from the same alignment (sanity check that
    del + ins + sub adds up; matches torchmetrics.text.WordErrorRate)."""

    def compute(self) -> torch.Tensor:
        return self._ratio(self.n_del + self.n_ins + self.n_sub)


if __name__ == "__main__":
    from torchmetrics import MetricCollection

    metrics = MetricCollection(
        {
            "del": DeletionRate(),
            "ins": InsertionRate(),
            "sub": SubstitutionRate(),
            "wer": WordErrorRateOps(),
        }
    )

    # Fig. 4 of the VAC paper: HYP_a has 2 deletions, HYP_p has 2 insertions.
    REF = "__ON__ HEUTE NACHT MEHR SCHNEE NORD SUEDOST ABER KALT".split()
    HYPa = "__ON__ HEUTE NACHT SCHNEE NORD SUEDOST ABER".split()
    HYPp = "__ON__ HEUTE NACHT MEHR SCHNEE NORD SUED SUEDOST SUED ABER KALT".split()

    metrics.update([REF], [HYPa])
    print("aux    :", {k: round(float(v), 4) for k, v in metrics.compute().items()})
    metrics.reset()
    metrics.update([REF], [HYPp])
    print("primary:", {k: round(float(v), 4) for k, v in metrics.compute().items()})
    print("compute groups:", metrics.compute_groups)

    # Cross-check against torchmetrics' own WER
    from torchmetrics.text import WordErrorRate

    tm_wer = WordErrorRate()
    tm_wer.update([" ".join(HYPp)], [" ".join(REF)])
    print("torchmetrics WER (primary):", round(float(tm_wer.compute()), 4))