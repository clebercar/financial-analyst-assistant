"""Testes do wrapper yfinance em src/data/collector.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.collector import baixar_dados_acao, extrair_preco_fechamento


def _fake_df(periods: int = 5) -> pd.DataFrame:
    """DataFrame fake no formato que yfinance retorna (MultiIndex de colunas)."""
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(periods)],
            "High": [101.0 + i for i in range(periods)],
            "Low": [99.0 + i for i in range(periods)],
            "Close": [100.5 + i for i in range(periods)],
            "Volume": [1_000_000 for _ in range(periods)],
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
    return df


def test_baixar_dados_acao_retorna_colunas_achatadas(monkeypatch):
    monkeypatch.setattr("yfinance.download", lambda *a, **kw: _fake_df())
    df = baixar_dados_acao("AAPL", "2024-01-01", "2024-01-05")
    assert not isinstance(df.columns, pd.MultiIndex)
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
    assert len(df) == 5


def test_baixar_dados_acao_remove_nas(monkeypatch):
    df_with_nan = _fake_df(periods=4)
    df_with_nan.iloc[1, :] = pd.NA
    monkeypatch.setattr("yfinance.download", lambda *a, **kw: df_with_nan)
    df = baixar_dados_acao("AAPL", "2024-01-01", "2024-01-04")
    assert len(df) == 3  # uma linha removida


def test_baixar_dados_acao_levanta_em_dataframe_vazio(monkeypatch):
    monkeypatch.setattr("yfinance.download", lambda *a, **kw: pd.DataFrame())
    with pytest.raises(ValueError, match="Nenhum dado encontrado"):
        baixar_dados_acao("INVALID")


def test_baixar_dados_acao_ordena_indice(monkeypatch):
    df = _fake_df()
    df = df.iloc[::-1]  # inverte a ordem
    monkeypatch.setattr("yfinance.download", lambda *a, **kw: df)
    out = baixar_dados_acao("AAPL", "2024-01-01", "2024-01-05")
    assert out.index.is_monotonic_increasing


def test_extrair_preco_fechamento_sucesso():
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0], "Volume": [1, 2, 3]})
    serie = extrair_preco_fechamento(df)
    assert isinstance(serie, pd.Series)
    assert len(serie) == 3
    assert serie.iloc[0] == 100.0


def test_extrair_preco_fechamento_levanta_sem_close():
    df = pd.DataFrame({"Open": [100.0]})
    with pytest.raises(ValueError, match="Close"):
        extrair_preco_fechamento(df)
