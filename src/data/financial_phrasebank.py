"""Loader do dataset FinancialPhraseBank via Hugging Face datasets.

O FinancialPhraseBank e um dataset de frases extraidas de noticias financeiras,
classificadas em 3 sentimentos por anotadores humanos. Tem subsets baseados no
nivel de concordancia entre os anotadores (50, 66, 75, 100%) - quanto maior a
concordancia, menor o tamanho mas maior a confiabilidade do label.

Usamos por padrao 'sentences_75agree' (~3.500 frases) - bom equilibrio entre
volume de dados e qualidade das anotacoes.
"""

import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Mapeamento dos labels numericos do dataset original para strings legiveis.
# A ordem e a convencao do FinancialPhraseBank: 0=negative, 1=neutral, 2=positive.
_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_phrasebank(config_name: str = "sentences_75agree") -> pd.DataFrame:
    """Carrega o FinancialPhraseBank em DataFrame com colunas sentence, label.

    Args:
        config_name: subset do dataset. Opcoes:
            - 'sentences_50agree' (~4.800 frases, 50% de concordancia)
            - 'sentences_66agree' (~4.200 frases)
            - 'sentences_75agree' (~3.500 frases) [DEFAULT]
            - 'sentences_allagree' (~2.300 frases, 100% de concordancia)

    Returns:
        DataFrame com colunas:
            - 'sentence' (str): texto da frase
            - 'label' (str): um de 'positive', 'neutral', 'negative'

    Notes:
        Primeira chamada baixa ~600MB de cache do HuggingFace. Chamadas
        subsequentes leem do cache local (~/.cache/huggingface).
    """
    logger.info("Carregando FinancialPhraseBank config=%s", config_name)
    ds = load_dataset("financial_phrasebank", config_name, trust_remote_code=True)
    # FinancialPhraseBank so tem split 'train' (sem test/val oficial - faremos split nos).
    df = ds["train"].to_pandas()

    # Se vier com label numerico, converte pra string. Se ja for string, deixa.
    if "label" in df.columns and df["label"].dtype.kind in ("i", "u"):
        df["label"] = df["label"].map(_LABEL_MAP)

    logger.info("Carregado %d frases", len(df))
    return df[["sentence", "label"]]
