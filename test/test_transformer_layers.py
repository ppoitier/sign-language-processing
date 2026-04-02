import torch
from torchtune.modules import RotaryPositionalEmbeddings

from slp.nn.blocks.transformers.layers import TransformerEncoderLayer, TransformerDecoderLayer


def test_transformer_encoder_inference():
    with torch.no_grad():
        N, T, C = 3, 128, 512
        x = torch.randn(N, T, C)
        layer = TransformerEncoderLayer(
            d_model=512,
            n_heads=2,
            dim_feedforward=2048,
            dropout=0.1,
            rope=None,
        )
        y = layer(x)
    assert not torch.isnan(y).any()
    assert x.shape == y.shape


def test_transformer_encoder_inference_with_rope():
    with torch.no_grad():
        N, T, C = 3, 128, 512
        N_HEADS = 2
        x = torch.randn(N, T, C)
        rope = RotaryPositionalEmbeddings(
            dim=C // N_HEADS,
        )
        layer = TransformerEncoderLayer(
            d_model=C,
            n_heads=N_HEADS,
            dim_feedforward=2048,
            dropout=0.1,
            rope=rope,
        )
        y = layer(x)
    assert not torch.isnan(y).any()
    assert x.shape == y.shape


def test_transformer_decoder_inference():
    with torch.no_grad():
        N, T_query, T_kv, C = 3, 4, 128, 512
        x_query = torch.randn(N, T_query, C)
        x_kv = torch.randn(N, T_kv, C)
        layer = TransformerDecoderLayer(
            d_model=512,
            n_heads=2,
            dim_feedforward=2048,
            dropout=0.1,
            rope=None,
        )
        y = layer(x_query, x_kv)
    assert not torch.isnan(y).any()
    assert x_query.shape == y.shape


def test_transformer_decoder_inference_with_rope():
    with torch.no_grad():
        N, T_query, T_kv, C = 3, 4, 128, 512
        N_HEADS = 2
        x_query = torch.randn(N, T_query, C)
        x_kv = torch.randn(N, T_kv, C)
        rope = RotaryPositionalEmbeddings(
            dim=C // N_HEADS,
        )
        layer = TransformerDecoderLayer(
            d_model=C,
            n_heads=N_HEADS,
            dim_feedforward=2048,
            dropout=0.1,
            rope=rope,
        )
        y = layer(x_query, x_kv)
    assert not torch.isnan(y).any()
    assert x_query.shape == y.shape
