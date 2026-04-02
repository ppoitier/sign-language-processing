import torch
from slp.nn.backbones.transformer import MultiStageTransformer


def test_ms_transformer_inference():
    model = MultiStageTransformer(
        in_channels=130,
        hidden_channels=512,
        max_length=1024,
        n_heads=4,
        n_stem_layers=2,
        n_branch_layers=4,
        dropout=0.1,
        pos_encoding='sinusoidal',
    )
    N, C, T = 3, 130, 1024

    with torch.no_grad():
        x = torch.randn(N, C, T)
        mask = torch.ones(N, 1, T).bool()
        out = model(x, mask)
    print([o.shape for o in out])
