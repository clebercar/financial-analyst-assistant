# Modulo responsavel por buscar dados historicos de acoes usando a API do Yahoo Finance.
# A ideia aqui e ter um unico lugar que sabe como baixar e limpar os dados brutos.

import pandas as pd
import yfinance as yf


def baixar_dados_acao(
    simbolo: str = "AAPL",
    data_inicio: str = "2018-01-01",
    data_fim: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Baixa o historico de precos de uma acao via Yahoo Finance.

    Parametros:
        simbolo: codigo da acao (ex: 'AAPL', 'PETR4.SA')
        data_inicio: data inicial no formato 'YYYY-MM-DD'
        data_fim: data final no formato 'YYYY-MM-DD'

    Retorna:
        DataFrame com colunas: Open, High, Low, Close, Volume
    """
    print(f"Baixando dados de {simbolo} de {data_inicio} ate {data_fim}...")

    df = yf.download(simbolo, start=data_inicio, end=data_fim, progress=False)

    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {simbolo}. Verifica se o simbolo ta certo.")

    # O yfinance as vezes retorna MultiIndex nas colunas quando baixa uma acao so.
    # Aqui a gente achata pra ficar simples de trabalhar.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove linhas com valores faltantes (feriados, dias sem pregao, etc.)
    df = df.dropna()

    # Garante que o indice e do tipo datetime e ta ordenado
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"Dados carregados: {len(df)} registros de {df.index[0].date()} ate {df.index[-1].date()}")

    return df


def extrair_preco_fechamento(df: pd.DataFrame) -> pd.Series:
    """
    Pega so a coluna de fechamento (Close), que e o que a gente vai usar pra treinar o modelo.
    O preco de fechamento e o mais relevante porque representa o valor final do dia.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame nao tem coluna 'Close'. Verifica se os dados foram baixados certo.")

    return df["Close"].copy()
