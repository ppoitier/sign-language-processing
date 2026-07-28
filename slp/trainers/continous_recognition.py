from torch import nn

from slp.trainers.generic import GenericTrainer
from slp.decoders.ctc import CTCDecoder, GreedyCTCDecoder


class ContinuousRecognitionTrainer(GenericTrainer):
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
        is_output_multistage: bool = False,
        classification_head: str = "classification",
        glosses_target: str = "glosses",
        ctc_decoder: CTCDecoder | None = None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            learning_rate=learning_rate,
            heads_to_targets=heads_to_targets,
            is_output_multistage=is_output_multistage,
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