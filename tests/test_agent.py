"""Testes unitarios do agente, tools e RAG. Tudo mockado (sem rede)."""

from unittest.mock import MagicMock


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
