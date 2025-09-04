from dataclasses import dataclass

from torch import nn, Tensor

from slp.nn.vqvae.quantizer import Quantizer
from slp.nn.functional import temporal_mse_loss


@dataclass
class TemporalVQVAEOutput:
    reconstructed_input: Tensor
    total_loss: Tensor
    reconstruction_loss: Tensor
    quantizer_loss: Tensor
    quantized_indices: Tensor


class TemporalVQVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        embedding_dim: int = 64,
        n_embeddings: int = 512,
        commitment_loss_factor: float = 0.25,
        quantization_loss_factor: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = Quantizer(
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            commitment_loss_factor=commitment_loss_factor,
            quantization_loss_factor=quantization_loss_factor,
        )

    def loss_function(
        self,
        original_input: Tensor,
        reconstructed_input: Tensor,
        quantizer_loss: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        reconstruction_loss = temporal_mse_loss(reconstructed_input, original_input, mask)
        total_loss = reconstruction_loss + quantizer_loss
        return total_loss, reconstruction_loss

    def forward(self, x: Tensor, mask: Tensor) -> TemporalVQVAEOutput:
        # 1. Encode the input image
        # Input: (B, T, *) -> Output: (B, T, E)
        z_e = self.encoder(x, mask=mask)

        # 2. Quantize the latent vectors
        quantizer_output = self.quantizer(z_e, mask=mask)

        # 4. Decode the quantized vectors to reconstruct the image
        # Input: (B, T, E) -> Output: (B, T, C_out)
        x_hat = self.decoder(quantizer_output.quantized_vectors, mask=mask)

        # 5. Calculate losses
        total_loss, recon_loss = self.loss_function(x, x_hat, quantizer_output.loss, mask=mask)

        return TemporalVQVAEOutput(
            reconstructed_input=x_hat,
            total_loss=total_loss,
            reconstruction_loss=recon_loss,
            quantizer_loss=quantizer_output.loss,
            quantized_indices=quantizer_output.quantized_indices,
        )
