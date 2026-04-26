"""Ferramentas (tools) do agente ReAct.

Este modulo expoe funcoes "tool-shaped" que o agente Gemini pode chamar.

Convencoes:
- Lazy load de modelos pesados (LSTM, sentiment) via variaveis modulo-globais
  para evitar travar a importacao do modulo se artefatos nao existirem.
- Sempre retornar dict serializavel (sem objetos numpy/pandas).
- Em erros nao-fatais, retornar `{"erro": "..."}` ao inves de levantar.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Lazy load dos modelos pra evitar travar a importacao se artefatos sumirem.
# As tools prever_preco_lstm e analisar_sentimento sao implementadas no Dia 5
# e usarao essas variaveis.
_LSTM_MODEL: Any = None
_LSTM_SCALER: Any = None
_SENTIMENT_PIPELINE: Any = None


def consultar_preco(ticker: str) -> dict:
    """Preco atual e variacao dos ultimos 30 dias.

    Args:
        ticker: simbolo da acao (ex: "AAPL").

    Returns:
        dict com chaves: ticker, preco_atual, moeda, variacao_30d_pct,
        volume_medio, timestamp (ISO 8601 UTC). Em erro: {"ticker": ..., "erro": "..."}.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="30d")
    if hist.empty:
        return {"ticker": ticker.upper(), "erro": "ticker nao encontrado ou sem dados"}
    preco_atual = float(hist["Close"].iloc[-1])
    preco_30d_atras = float(hist["Close"].iloc[0])
    variacao = (preco_atual - preco_30d_atras) / preco_30d_atras * 100
    volume_medio = float(hist["Volume"].mean())
    return {
        "ticker": ticker.upper(),
        "preco_atual": round(preco_atual, 2),
        "moeda": "USD",
        "variacao_30d_pct": round(variacao, 2),
        "volume_medio": round(volume_medio, 0),
        "timestamp": datetime.now(UTC).isoformat(),
    }
