from dataclasses import dataclass

import torch
from torch import nn, Tensor

from slp.nn.functional import temporal_mse_loss


@dataclass
class QuantizerOutput:
    quantized_vectors: torch.Tensor
    quantized_indices: torch.Tensor
    loss: torch.Tensor


class Quantizer(nn.Module):
    def __init__(
            self,
            n_embeddings: int,
            embedding_dim: int,
            commitment_loss_factor: float,
            quantization_loss_factor: float,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss_factor = commitment_loss_factor
        self.quantization_loss_factor = quantization_loss_factor

        self.embeddings = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.n_embeddings, 1 / self.n_embeddings
        )

    def forward(self, z: Tensor, mask: Tensor):
        # Input z is expected to be (B, *, E) from the encoder
        original_shape = z.shape
        # (B, *, E) -> (*, E)
        z_flat = z.flatten(end_dim=-2)

        # --- Find nearest neighbors ---
        # Calculate the squared Euclidean distance between input vectors and embeddings
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )

        # Find the closest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        quantized_flat = self.embeddings(encoding_indices)
        quantized = quantized_flat.reshape(original_shape).contiguous()

        # --- Calculate losses ---
        # The embedding loss (or codebook loss) updates the embedding vectors to match the encoder's output.
        embedding_loss = temporal_mse_loss(quantized, z.detach(), mask)
        # The commitment loss updates the encoder to produce outputs closer to the chosen embedding.
        commitment_loss = temporal_mse_loss(z, quantized.detach(), mask)

        loss = (
            self.quantization_loss_factor * embedding_loss
            + self.commitment_loss_factor * commitment_loss
        )

        # --- Straight-Through Estimator (STE) ---
        # Copy gradients from `quantized` to `z` in the backward pass.
        quantized = z + (quantized - z).detach()

        # --- Prepare output ---
        quantized_vectors = quantized
        quantized_indices = encoding_indices.reshape(original_shape[:-1])

        return QuantizerOutput(
            quantized_vectors=quantized_vectors,
            quantized_indices=quantized_indices,
            loss=loss,
        )
