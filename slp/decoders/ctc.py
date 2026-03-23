from abc import ABC, abstractmethod

import torch
from torch import Tensor


class CTCDecoder(ABC):
    """Decodes frame-level logits into a sequence of token IDs via CTC logic."""

    @abstractmethod
    def decode(self, log_probs: Tensor) -> list[int]:
        """Decode a single instance.

        Args:
            log_probs: (C, T) log-probabilities over the vocabulary.

        Returns:
            List of predicted token IDs (blanks and duplicates removed).
        """
        ...

    def decode_batch(self, log_probs: Tensor) -> list[list[int]]:
        """Decode each instance in a batch.

        Args:
            log_probs: (B, C, T) log-probabilities.

        Returns:
            List of predicted token ID sequences, one per instance.
        """
        return [self.decode(log_probs[i]) for i in range(log_probs.size(0))]


class GreedyCTCDecoder(CTCDecoder):
    """Greedy (best-path) CTC decoding.

    Takes the argmax at each timestep, then collapses repeated tokens
    and removes blanks.

    Args:
        blank_id: Index of the CTC blank token.
    """

    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    def decode(self, log_probs: Tensor) -> list[int]:
        # log_probs: (C, T)
        best_path = log_probs.argmax(dim=0).tolist()  # (T,)

        decoded = []
        prev = None
        for token in best_path:
            if token != self.blank_id and token != prev:
                decoded.append(token)
            prev = token
        return decoded


class BeamSearchCTCDecoder(CTCDecoder):
    """Beam search CTC decoding using torchaudio.

    Falls back to greedy decoding if torchaudio is not available.

    Args:
        blank_id: Index of the CTC blank token.
        beam_width: Number of beams to keep.
    """

    def __init__(self, blank_id: int = 0, beam_width: int = 10):
        self.blank_id = blank_id
        self.beam_width = beam_width
        self._greedy_fallback = GreedyCTCDecoder(blank_id=blank_id)

        try:
            from torchaudio.models.decoder import ctc_decoder
            self._decoder = ctc_decoder
            self._available = True
        except ImportError:
            self._available = False

    def decode(self, log_probs: Tensor) -> list[int]:
        if not self._available:
            return self._greedy_fallback.decode(log_probs)

        # torchaudio expects (T, C)
        log_probs_t = log_probs.T.unsqueeze(0).cpu()
        # Minimal setup — no language model
        results = self._decoder(
            blank_token="<blank>",
            tokens=[str(i) for i in range(log_probs.size(0))],
            beam_size=self.beam_width,
        )(log_probs_t)
        if results and results[0]:
            return results[0][0].tokens.tolist()
        return self._greedy_fallback.decode(log_probs)