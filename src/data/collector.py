"""Coleta de dados historicos de acoes via Yahoo Finance.

Wrapper fino sobre `yfinance.download` com:
- normalizacao de MultiIndex (yfinance retorna colunas em multi-level quando
  baixa um unico ticker);
- limpeza de linhas com NaN (feriados, dias sem pregao);
- ordenacao do indice temporal.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def baixar_dados_acao(
    simbolo: str = "AAPL",
    data_inicio: str = "2018-01-01",
    data_fim: str = "2024-12-31",
) -> pd.DataFrame:
    """Baixa o historico de precos de uma acao via Yahoo Finance.

    Args:
        simbolo: codigo da acao (ex: 'AAPL', 'PETR4.SA').
        data_inicio: data inicial no formato 'YYYY-MM-DD'.
        data_fim: data final no formato 'YYYY-MM-DD'.

    Returns:
        DataFrame com colunas Open, High, Low, Close, Volume e indice
        datetime ordenado.

    Raises:
        ValueError: se o yfinance retornar dataframe vazio (simbolo invalido
            ou periodo sem dados).
    """
    logger.info("Baixando dados de %s de %s ate %s", simbolo, data_inicio, data_fim)

    df = yf.download(simbolo, start=data_inicio, end=data_fim, progress=False)

    if df.empty:
        raise ValueError(
            f"Nenhum dado encontrado para {simbolo}. Verifique se o simbolo esta correto."
        )

    # yfinance retorna MultiIndex nas colunas quando baixa um unico ticker.
    # Achatamos para o nivel superior (Open, High, Low, Close, Volume).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove linhas com valores faltantes (feriados, dias sem pregao).
    df = df.dropna()

    # Garante indice datetime ordenado.
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    logger.info(
        "Dados carregados: %d registros de %s ate %s",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )

    return df


def extrair_preco_fechamento(df: pd.DataFrame) -> pd.Series:
    """Extrai a coluna de fechamento (Close) do DataFrame.

    O preco de fechamento e o valor de referencia diario para previsao de
    series temporais.

    Raises:
        ValueError: se a coluna 'Close' nao existir.
    """
    if "Close" not in df.columns:
        raise ValueError(
            "DataFrame nao tem coluna 'Close'. Verifique se os dados foram baixados corretamente."
        )

    return df["Close"].copy()
