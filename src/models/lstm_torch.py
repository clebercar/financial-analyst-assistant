"""LSTM em PyTorch para previsao de precos de acoes.

Conversao da arquitetura Keras da Fase 4 mantendo a mesma topologia:
    LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(25, ReLU) -> Dense(1)

Por que portar pra PyTorch?
- O Datathon Fase 5 padronizou o stack em PyTorch + MLflow (PyTorch tem hook nativo
  no MLflow e suporta tracing/logging mais granular).
- Mantemos a topologia identica pra que o desempenho seja comparavel a Fase 4
  (mesma quantidade de parametros, mesmas regularizacoes).

Nota sobre num_layers:
- Em PyTorch, dropout="entre camadas" e aplicado pelo proprio nn.LSTM quando
  num_layers > 1. Por isso usamos num_layers=2 ao inves de duas camadas
  separadas, mas o efeito e equivalente a duas LSTM(50) empilhadas com Dropout
  entre elas.
"""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM bicamada para regressao de preco de fechamento.

    Recebe sequencias de shape (batch, seq_len, input_size) e retorna
    um tensor de shape (batch, 1) com o valor previsto.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        dense_size: int = 25,
    ) -> None:
        super().__init__()
        # nn.LSTM com num_layers > 1 ja aplica Dropout entre as camadas LSTM.
        # batch_first=True facilita o trabalho - shape de entrada vira (batch, seq, feat).
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Dropout extra na saida do LSTM antes da camada densa (equivalente ao
        # segundo Dropout(0.2) que tinha na arquitetura Keras).
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, dense_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(dense_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass do modelo.

        Args:
            x: tensor de shape (batch, seq_len, input_size)

        Returns:
            tensor de shape (batch, 1) com o valor previsto pra cada amostra.
        """
        # lstm_out: (batch, seq_len, hidden_size). Cada timestep tem o estado
        # oculto correspondente. Como queremos prever 1 valor depois da
        # sequencia inteira, pegamos o ultimo timestep (equivalente a
        # return_sequences=False na ultima LSTM da Keras).
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        x = self.dropout(last)
        x = self.activation(self.dense(x))
        return self.output(x)
