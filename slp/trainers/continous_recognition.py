from torch import nn

from slp.trainers.generic import GenericTrainer
from slp.decoders.ctc import CTCDecoder, GreedyCTCDecoder


def compute_wer(predicted: list[list[int]], references: list[list[int]]) -> dict:
    """Compute Word Error Rate using edit distance.

    Args:
        predicted: List of predicted token ID sequences.
        references: List of reference token ID sequences.

    Returns:
        Dict with 'substitutions', 'deletions', 'insertions',
        'total_ref_tokens', and 'wer'.
    """
    total_sub, total_del, total_ins, total_ref = 0, 0, 0, 0

    for pred, ref in zip(predicted, references):
        n, m = len(ref), len(pred)
        # Standard edit distance DP with operation tracking
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i - 1] == pred[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j - 1],  # substitution
                        dp[i - 1][j],       # deletion
                        dp[i][j - 1],       # insertion
                    )

        # Backtrace to count S, D, I
        i, j = n, m
        s, d, ins = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i - 1] == pred[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                s += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                d += 1
                i -= 1
            else:
                ins += 1
                j -= 1

        total_sub += s
        total_del += d
        total_ins += ins
        total_ref += n

    wer = (total_sub + total_del + total_ins) / max(total_ref, 1)
    return {
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "total_ref_tokens": total_ref,
        "wer": wer,
    }


class RecognitionTrainer(GenericTrainer):
    """Trainer for continuous sign language recognition (CTC-based).

    Computes WER by decoding frame-level logits into gloss sequences
    and comparing against ground-truth token sequences.

    The criterion (e.g. CTCLoss) is owned by GenericTrainer. This subclass
    only adds CTC decoding and WER evaluation.

    Args:
        model: The model to train.
        criterion: Loss function (typically wrapping nn.CTCLoss).
        learning_rate: Optimizer learning rate.
        heads_to_targets: Mapping from model head names to target keys.
        is_output_multilayer: Whether the model returns multi-layer outputs.
        classification_head: Key for the recognition logits in the model output.
        glosses_target: Key for ground-truth gloss ID sequences in batch targets.
        ctc_decoder: Decoder for converting logits to token sequences.
            Defaults to greedy decoding.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        heads_to_targets: dict[str, str],
        is_output_multilayer: bool = False,
        classification_head: str = "classification",
        glosses_target: str = "glosses",
        ctc_decoder: CTCDecoder | None = None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            learning_rate=learning_rate,
            heads_to_targets=heads_to_targets,
            is_output_multilayer=is_output_multilayer,
        )
        self.classification_head = classification_head
        self.glosses_target = glosses_target
        self.ctc_decoder = ctc_decoder or GreedyCTCDecoder()

        self.save_hyperparameters(ignore=["model", "criterion"])

    def compute_metrics(self, logits: dict, batch: dict, mode: str) -> dict:
        cls_logits = logits[self.classification_head]
        log_probs = cls_logits.log_softmax(dim=1)  # (B, C, T)

        predicted = self.ctc_decoder.decode_batch(log_probs)
        references = batch["targets"][self.glosses_target]  # list of list[int]

        wer_results = compute_wer(predicted, references)
        return {
            f"{mode}/wer": wer_results["wer"],
            f"{mode}/substitutions": wer_results["substitutions"],
            f"{mode}/deletions": wer_results["deletions"],
            f"{mode}/insertions": wer_results["insertions"],
        }