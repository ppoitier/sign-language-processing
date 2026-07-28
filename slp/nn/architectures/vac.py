from torch import nn, Tensor


class VacModel(nn.Module):
    def __init__(
            self,
            visual_encoder: nn.Module,
            c_out_visual: int,
            contextual_encoder: nn.Module,
            c_out_contextual: int,
            n_classes: int,
    ):
        super().__init__()
        self.visual = visual_encoder
        self.visual_head = nn.Conv1d(c_out_visual, n_classes, 1)
        self.contextual = contextual_encoder
        self.contextual_head = nn.Conv1d(c_out_contextual, n_classes, 1)

    def forward(self, x: Tensor, mask: Tensor) -> dict[str, Tensor]:
        visual_embeddings = self.visual(x)
        contextual_embeddings = self.contextual(visual_embeddings)

        visual_logits = self.visual_head(visual_embeddings)
        contextual_logits = self.contextual_head(contextual_embeddings)

        return {
            'visual_embeddings': visual_embeddings,
            'visual_logits': visual_logits,
            'contextual_embeddings': contextual_embeddings,
            'contextual_logits': contextual_logits,
        }