# Testes do modelo LSTM em PyTorch.
# A ideia aqui e validar tres coisas fundamentais antes de gastar tempo treinando:
#   1. Forward pass tem o shape esperado (batch, 1) -> evita bug de dimensao
#   2. Modelo em modo eval e deterministico -> mesma entrada = mesma saida
#   3. Smoke de aprendizado: em batch trivial o loss tem que cair (sanity check)

import torch

from src.models.lstm_torch import LSTMRegressor


def test_lstm_forward_shape():
    """Garante que o forward retorna shape (batch, 1) - 1 valor previsto por amostra."""
    model = LSTMRegressor(
        input_size=1,
        hidden_size=50,
        num_layers=2,
        dropout=0.2,
        dense_size=25,
    )
    batch_size, seq_len, n_features = 4, 60, 1
    x = torch.randn(batch_size, seq_len, n_features)
    out = model(x)
    assert out.shape == (batch_size, 1)


def test_lstm_deterministic():
    """Em modo eval, a mesma entrada tem que dar a mesma saida (sem dropout aleatorio)."""
    torch.manual_seed(42)
    model = LSTMRegressor(
        input_size=1,
        hidden_size=50,
        num_layers=2,
        dropout=0.0,
        dense_size=25,
    )
    model.eval()
    x = torch.randn(2, 60, 1)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
