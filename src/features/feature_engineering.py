"""Schemas de validacao e funcoes de feature engineering.

Pandera nos da validacao declarativa de DataFrame: declaramos o schema
esperado (colunas, tipos, restricoes) e ele falha cedo se o dado nao bater.
Vantagem em pipeline ML: bug de label invalido aparece ANTES do treino, nao
depois de 30min esperando o classifier convergir em lixo.
"""

import pandas as pd
import pandera.pandas as pa  # noqa: F401  (referencia futura para outros schemas)
from pandera.pandas import Check, Column, DataFrameSchema

# Dominio fechado de labels - tudo o que nao for um destes 3 e considerado invalido.
VALID_LABELS = ["positive", "neutral", "negative"]

# Schema do input do treino de sentimento.
# strict=False: permite colunas extras no DataFrame (ex: id, source). Apenas
# garantimos que sentence e label estao la, com os tipos certos e dentro do
# dominio.
SENTIMENT_INPUT_SCHEMA = DataFrameSchema(
    {
        "sentence": Column(str, Check.str_length(min_value=1)),
        "label": Column(str, Check.isin(VALID_LABELS)),
    },
    strict=False,
)


def validate_sentiment_input(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame de entrada do treino de sentimento.

    Args:
        df: DataFrame que deve ter colunas 'sentence' (str nao vazia) e
            'label' (str em VALID_LABELS).

    Returns:
        Mesmo DataFrame, ja validado. Pandera retorna o objeto pois pode ter
        coercao de tipo opcional (nao e o caso aqui, mas mantemos o padrao).

    Raises:
        pandera.errors.SchemaError: se alguma checagem falhar.
    """
    return SENTIMENT_INPUT_SCHEMA.validate(df)
