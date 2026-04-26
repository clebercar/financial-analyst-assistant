"""Testes unitarios do agente, tools e RAG. Tudo mockado (sem rede)."""

from unittest.mock import MagicMock

import numpy as np


def test_chunking_splits_long_text():
    """Texto longo deve ser quebrado em multiplos chunks com tamanho controlado."""
    from src.agent.rag_pipeline import chunk_text

    text = "Este e um teste. " * 500  # ~8500 chars
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # chunk_size=200 tokens * 4 = 800 chars; tolerancia de overlap
    assert all(len(c) <= 200 * 4 + 1 for c in chunks)


def test_chunking_short_text_returns_single_chunk():
    """Texto curto deve retornar um unico chunk."""
    from src.agent.rag_pipeline import chunk_text

    chunks = chunk_text("texto curto", chunk_size=100, overlap=10)
    assert len(chunks) == 1


def test_retrieve_calls_collection_query():
    """retrieve() deve chamar collection.query e formatar a saida em dicts."""
    from src.agent.rag_pipeline import retrieve

    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [
            [
                {"ticker": "AAPL", "filing_type": "10-K", "year": "2024"},
                {"ticker": "MSFT", "filing_type": "10-Q", "year": "2024"},
            ]
        ],
        "distances": [[0.1, 0.3]],
    }
    fake_embed_fn = MagicMock(return_value=[0.0] * 768)
    chunks = retrieve("query", fake_collection, fake_embed_fn, top_k=2)
    assert len(chunks) == 2
    assert chunks[0]["trecho"] == "doc1"
    assert chunks[0]["ticker"] == "AAPL"
    assert chunks[0]["tipo"] == "10-K"
    assert chunks[0]["distance"] == 0.1


def test_consultar_preco_basico(monkeypatch):
    """consultar_preco deve retornar dict com preco, variacao e timestamp ISO."""
    import pandas as pd

    from src.agent.tools import consultar_preco

    fake_hist = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=pd.date_range("2024-01-01", periods=5),
    )

    class FakeTicker:
        def history(self, period):  # noqa: ARG002
            return fake_hist

    monkeypatch.setattr("yfinance.Ticker", lambda t: FakeTicker())  # noqa: ARG005
    result = consultar_preco("AAPL")
    assert result["ticker"] == "AAPL"
    assert "preco_atual" in result
    assert "variacao_30d_pct" in result
    assert result["preco_atual"] == 104.0
    # Variacao: (104-100)/100*100 = 4.0%
    assert result["variacao_30d_pct"] == 4.0
    assert "timestamp" in result
    # ISO 8601 deve poder ser parseado
    from datetime import datetime

    datetime.fromisoformat(result["timestamp"])


def test_prever_preco_lstm_aapl(monkeypatch):
    """prever_preco_lstm com ticker AAPL deve retornar previsoes sem aviso de dominio."""
    import torch

    from src.agent import tools as tools_mod
    from src.agent.tools import prever_preco_lstm

    class FakeModel:
        def eval(self):
            pass

        def __call__(self, x):  # noqa: ARG002
            return torch.tensor([[0.5]])

    class FakeScaler:
        def transform(self, x):
            return x

        def inverse_transform(self, x):
            return x * 200  # escala fake

    monkeypatch.setattr(tools_mod, "_get_lstm_model", lambda: (FakeModel(), FakeScaler()))
    monkeypatch.setattr(
        tools_mod,
        "_get_recent_prices",
        lambda ticker, days=60: np.linspace(100, 110, 60),  # noqa: ARG005
    )

    result = prever_preco_lstm("AAPL", dias=3)
    assert result["ticker"] == "AAPL"
    assert len(result["previsoes"]) == 3
    # Cada previsao tem dia (1..N) e preco_previsto
    assert {"dia", "preco_previsto"} <= set(result["previsoes"][0].keys())
    # AAPL e o ticker treinado: aviso vazio
    assert result["aviso"] == ""


def test_prever_preco_lstm_outro_ticker_avisa(monkeypatch):
    """Tickers != AAPL recebem aviso explicito de dominio fora-da-distribuicao."""
    import torch

    from src.agent import tools as tools_mod
    from src.agent.tools import prever_preco_lstm

    class FakeModel:
        def eval(self):
            pass

        def __call__(self, x):  # noqa: ARG002
            return torch.tensor([[0.5]])

    class FakeScaler:
        def transform(self, x):
            return x

        def inverse_transform(self, x):
            return x * 200

    monkeypatch.setattr(tools_mod, "_get_lstm_model", lambda: (FakeModel(), FakeScaler()))
    monkeypatch.setattr(
        tools_mod,
        "_get_recent_prices",
        lambda ticker, days=60: np.linspace(200, 210, 60),  # noqa: ARG005
    )

    result = prever_preco_lstm("MSFT", dias=2)
    assert "treinado apenas em AAPL" in result.get("aviso", "")
    assert result["ticker"] == "MSFT"


def test_analisar_sentimento(monkeypatch):
    """analisar_sentimento deve usar o pipeline mockado e retornar dict {sentimento, confianca}."""
    from src.agent import tools as tools_mod
    from src.agent.tools import analisar_sentimento

    pipe = MagicMock()
    pipe.predict.return_value = ["positive"]
    pipe.predict_proba.return_value = [[0.05, 0.10, 0.85]]
    pipe.classes_ = ["negative", "neutral", "positive"]
    monkeypatch.setattr(tools_mod, "_get_sentiment_pipeline", lambda: pipe)

    result = analisar_sentimento("Earnings beat expectations")
    assert result["sentimento"] == "positive"
    assert result["confianca"] > 0.8


def test_buscar_em_filings(monkeypatch):
    """buscar_em_filings deve delegar pra _rag_retrieve e empacotar resultado."""
    from src.agent import tools as tools_mod
    from src.agent.tools import buscar_em_filings

    fake_chunks = [
        {
            "ticker": "AAPL",
            "tipo": "10-K",
            "ano": "2024",
            "secao": "Risk Factors",
            "trecho": "...",
            "distance": 0.2,
        }
    ]
    monkeypatch.setattr(tools_mod, "_rag_retrieve", lambda q, t, k: fake_chunks)  # noqa: ARG005

    result = buscar_em_filings("Apple risks", ticker="AAPL", top_k=1)
    assert result["chunks"] == fake_chunks
    assert result["total_encontrado"] == 1


def test_buscar_em_filings_sem_filtro_ticker(monkeypatch):
    """buscar_em_filings sem ticker deve passar None pro retrieve."""
    from src.agent import tools as tools_mod
    from src.agent.tools import buscar_em_filings

    captured = {}

    def fake_retrieve(query, ticker, top_k):  # noqa: ARG001
        captured["ticker"] = ticker
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(tools_mod, "_rag_retrieve", fake_retrieve)

    result = buscar_em_filings("query qualquer", top_k=2)
    assert captured["ticker"] is None
    assert captured["top_k"] == 2
    assert result["total_encontrado"] == 0
