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

import joblib
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
import yfinance as yf

logger = logging.getLogger(__name__)

# Lazy load dos modelos pra evitar travar a importacao se artefatos sumirem.
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


def _get_lstm_model() -> tuple[Any, Any]:
    """Carrega LSTM PyTorch + scaler de forma lazy (uma unica vez por processo).

    Returns:
        Tupla (model, scaler) com model em modo eval. Levanta FileNotFoundError
        se os artefatos nao existirem.
    """
    global _LSTM_MODEL, _LSTM_SCALER
    if _LSTM_MODEL is None:
        from src.models.lstm_torch import LSTMRegressor

        with open("configs/model_config.yaml") as f:
            cfg = yaml.safe_load(f)["lstm"]
        model = LSTMRegressor(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            dense_size=cfg["dense_size"],
        )
        model.load_state_dict(torch.load("models/lstm_torch.pt", map_location="cpu"))
        model.eval()
        _LSTM_MODEL = model
        _LSTM_SCALER = joblib.load("models/scaler.joblib")
    return _LSTM_MODEL, _LSTM_SCALER


def _get_recent_prices(ticker: str, days: int = 60) -> np.ndarray:
    """Busca os ultimos `days` precos de fechamento via yfinance.

    Pegamos um pouco mais que `days` no periodo e cortamos no final pra garantir
    que datas de feriado/fim de semana nao deixem a amostra menor que o esperado.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{days + 30}d")
    return hist["Close"].values[-days:]


def prever_preco_lstm(ticker: str, dias: int = 5) -> dict:
    """Preve N proximos dias uteis usando o LSTM treinado.

    O modelo foi treinado APENAS com dados da AAPL. Para outros tickers,
    a previsao funciona tecnicamente mas o modelo nao aprendeu o padrao
    daquele ativo - por isso retornamos um aviso explicito no campo `aviso`.

    Args:
        ticker: simbolo da acao.
        dias: quantos dias uteis prever (recursivamente, alimentando a previsao
            do dia anterior como entrada do proximo).

    Returns:
        dict com ticker, previsoes (lista de {dia, preco_previsto}),
        metricas_modelo, modelo_versao, aviso. Em erro: {"ticker": ..., "erro": "..."}.
    """
    aviso = (
        ""
        if ticker.upper() == "AAPL"
        else (
            "Aviso: modelo treinado apenas em AAPL - " "previsao para outros tickers e experimental"
        )
    )
    try:
        model, scaler = _get_lstm_model()
    except FileNotFoundError:
        return {
            "ticker": ticker.upper(),
            "erro": "modelo LSTM nao encontrado - rode `make train-lstm`",
            "aviso": aviso,
        }

    precos = np.asarray(_get_recent_prices(ticker, days=60)).reshape(-1, 1)
    if len(precos) < 60:
        return {
            "ticker": ticker.upper(),
            "erro": f"precisa 60 dias de historico, encontrado {len(precos)}",
            "aviso": aviso,
        }

    precos_norm = scaler.transform(precos)
    previsoes: list[dict] = []
    seq = precos_norm.copy()
    for d in range(dias):
        x = torch.from_numpy(seq[-60:].reshape(1, 60, 1)).float()
        with torch.no_grad():
            pred_norm = model(x)
        # pred_norm pode ser tensor ou ndarray (mock no teste); normalizamos pra ndarray.
        pred_norm_np = pred_norm.numpy() if hasattr(pred_norm, "numpy") else np.asarray(pred_norm)
        pred = float(scaler.inverse_transform(pred_norm_np)[0][0])
        previsoes.append({"dia": d + 1, "preco_previsto": round(pred, 2)})
        seq = np.vstack([seq, pred_norm_np.reshape(-1, 1)])

    return {
        "ticker": ticker.upper(),
        "previsoes": previsoes,
        "metricas_modelo": {"mae": "ver MLflow", "rmse": "ver MLflow"},
        "modelo_versao": "0.1.0",
        "aviso": aviso,
    }


def _get_sentiment_pipeline() -> Any:
    """Carrega o classificador de sentimento (sklearn Pipeline) lazy."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = joblib.load("models/sentiment_classifier.joblib")
    return _SENTIMENT_PIPELINE


def analisar_sentimento(texto: str) -> dict:
    """Classifica sentimento de um trecho financeiro (positive/neutral/negative).

    Args:
        texto: trecho de texto em ingles (modelo treinado em FinancialPhraseBank).

    Returns:
        dict com sentimento (str) e confianca (float em [0, 1]). Em erro:
        {"erro": "..."}.
    """
    try:
        pipe = _get_sentiment_pipeline()
    except FileNotFoundError:
        return {"erro": "classificador nao treinado - rode `make train-sentiment`"}
    from src.models.sentiment_classifier import predict_sentiment

    return predict_sentiment(texto, pipe)
