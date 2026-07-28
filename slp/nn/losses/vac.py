"""Visual Alignment Constraint (VAC) loss for CSLR.

Reference: Min et al., "Visual Alignment Constraint for Continuous Sign
Language Recognition", ICCV 2021 (arXiv:2104.02330).

Total loss (Eq. 6):  L = L_CTC + L_VE + alpha * L_VA
  - L_CTC: CTC on contextual logits (after contextual module, classifier F_p)
  - L_VE : CTC on visual logits (auxiliary classifier F_a on visual features)
  - L_VA : KD loss, teacher = contextual logits, student = visual logits (Eq. 5)

Expected `outputs` dict (adapt keys to your pipeline):
  "visual_logits"  : (N, C, T)  logits from F_a on framewise visual features
  "context_logits" : (N, C, T)  logits from F_p after the contextual module
  "logit_lengths"  : (N,)       valid time steps per sample (after any temporal
                                downsampling in your backbone)

  Here C is the number of CTC classes (gloss vocabulary + blank), NOT the
  feature width: feed logits AFTER the F_p / F_a classifier heads.

Expected `targets` dict:
  "glosses"        : (N, L_max) padded gloss index sequences (no blanks)
  "gloss_lengths"  : (N,)       true lengths of each gloss sequence

Notes:
  * Blank index defaults to 0; make sure your vocabulary reserves it.
  * Logits are channels-second (N, C, T); they are permuted to the time-first
    (T, N, C) layout that CTCLoss and the KD softmaxes expect.
  * If your model applies log_softmax already, pass raw logits instead --
    this module applies the softmaxes itself.
  * The VE loss only reaches the feature extractor + F_a by construction
    (visual logits never pass through the contextual module), matching the
    paper's theta^v restriction without any manual gradient surgery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VACLoss(nn.Module):
    def __init__(
        self,
        blank: int = 0,
        alpha: float = 25.0,
        temperature: float = 8.0,
        detach_teacher: bool = True,
        scale_kd_by_tau2: bool = True,
    ):
        """
        Args:
            blank: CTC blank index.
            alpha: weight of the VA (distillation) loss. Paper: 25.
            temperature: KD softening temperature tau. Paper: 8.
            detach_teacher: stop gradients through the contextual (teacher)
                logits in the VA loss. Keeps VA acting as supervision *for the
                visual branch* rather than dragging the teacher toward the
                student. Set False to reproduce a fully symmetric variant.
            scale_kd_by_tau2: multiply KD term by tau^2 (standard Hinton
                scaling so gradient magnitude is ~independent of tau). If you
                disable it, you are effectively just rescaling alpha.
        """
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        self.alpha = alpha
        self.tau = temperature
        self.detach_teacher = detach_teacher
        self.scale_kd_by_tau2 = scale_kd_by_tau2

    def forward(self, outputs: dict, targets: dict) -> dict:
        zv = outputs["visual_logits"].permute(2, 0, 1)   # (N,C,T) -> (T,N,C)
        zc = outputs["context_logits"].permute(2, 0, 1)  # (N,C,T) -> (T,N,C)
        in_lens = outputs["logit_lengths"]               # (N,)
        glosses = targets["glosses"]                     # (N, L_max) padded
        tgt_lens = targets["gloss_lengths"]              # (N,)

        # ---- primary CTC loss on contextual predictions -------------------
        l_ctc = self.ctc(zc.log_softmax(dim=-1), glosses, in_lens, tgt_lens)

        # ---- VE loss: CTC directly on the visual branch --------------------
        l_ve = self.ctc(zv.log_softmax(dim=-1), glosses, in_lens, tgt_lens)

        # ---- VA loss: distill contextual (teacher) into visual (student) ---
        teacher = zc.detach() if self.detach_teacher else zc
        log_p_student = (zv / self.tau).log_softmax(dim=-1)
        p_teacher = (teacher / self.tau).softmax(dim=-1)

        # Mask padded frames so they don't contribute to the KD term.
        T, N, _ = zv.shape
        frame_mask = (
            torch.arange(T, device=zv.device)[:, None] < in_lens.to(zv.device)[None, :]
        )  # (T, N)
        kd_per_frame = F.kl_div(
            log_p_student, p_teacher, reduction="none"
        ).sum(dim=-1)                            # (T, N)
        l_va = (kd_per_frame * frame_mask).sum() / frame_mask.sum().clamp(min=1)
        if self.scale_kd_by_tau2:
            l_va = l_va * (self.tau ** 2)

        loss = l_ctc + l_ve + self.alpha * l_va
        return {"loss": loss, "loss_ctc": l_ctc, "loss_ve": l_ve, "loss_va": l_va}


if __name__ == "__main__":
    # Smoke test -- logits are now (N, C, T)
    N, C, T, L = 2, 30, 50, 8
    outputs = {
        "visual_logits": torch.randn(N, C, T, requires_grad=True),
        "context_logits": torch.randn(N, C, T, requires_grad=True),
        "logit_lengths": torch.tensor([50, 42]),
    }
    targets = {
        "glosses": torch.randint(1, C, (N, L)),
        "gloss_lengths": torch.tensor([8, 5]),
    }
    crit = VACLoss(blank=0)
    out = crit(outputs, targets)
    out["loss"].backward()
    print({k: round(float(v), 4) for k, v in out.items()})