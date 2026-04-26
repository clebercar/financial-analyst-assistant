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


def test_lstm_one_step_overfits_tiny_batch():
    """Smoke: em 1 batch pequeno o modelo tem que reduzir o loss em poucas epochs.

    Se isso falhar, tem algo errado no forward/backward (gradiente nao flui,
    detach indevido, etc.). E um teste barato que pega bugs grosseiros antes
    do treino real (que demora minutos).
    """
    torch.manual_seed(0)
    model = LSTMRegressor(hidden_size=10, num_layers=1, dropout=0.0, dense_size=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    x = torch.randn(8, 60, 1)
    y = torch.randn(8, 1)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Modelo nao esta aprendendo em batch trivial"


def test_sentiment_predict_returns_label():
    """Smoke: predict_sentiment usa o classifier injetado e devolve dict valido.

    Mockamos o pipeline sklearn pra evitar carregar modelo real / ler disco.
    Garantimos apenas o contrato da funcao: chave 'sentimento' e 'confianca'.
    """
    from unittest.mock import MagicMock

    from src.models.sentiment_classifier import predict_sentiment

    mock_clf = MagicMock()
    mock_clf.predict.return_value = ["positive"]
    mock_clf.predict_proba.return_value = [[0.05, 0.10, 0.85]]
    mock_clf.classes_ = ["negative", "neutral", "positive"]
    result = predict_sentiment("Earnings beat expectations", mock_clf)
    assert result["sentimento"] in ("positive", "neutral", "negative")
    assert 0.0 <= result["confianca"] <= 1.0
