# Testes de validacao de schema do input do classificador de sentimento.
# Pandera levanta erro quando o DataFrame nao bate com o schema declarado;
# usamos isso pra falhar cedo se o dataset vier corrompido ou com label
# fora do dominio esperado.

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.features.feature_engineering import (
    SENTIMENT_INPUT_SCHEMA,  # noqa: F401  (importado pra garantir que o simbolo existe)
    validate_sentiment_input,
)

# Pandera pode levantar SchemaError ou SchemaErrors (varios erros agregados)
# dependendo de como a validacao foi chamada. Aceitamos os dois no teste.
_PANDERA_ERRORS = (SchemaError, SchemaErrors)


def test_valid_input_passes():
    """DataFrame com colunas corretas e labels validos passa pela validacao."""
    df = pd.DataFrame(
        {
            "sentence": ["foo", "bar"],
            "label": ["positive", "neutral"],
        }
    )
    validated = validate_sentiment_input(df)
    assert len(validated) == 2


def test_invalid_label_fails():
    """Label fora do dominio (positive/neutral/negative) deve quebrar a validacao."""
    df = pd.DataFrame(
        {
            "sentence": ["foo"],
            "label": ["INVALID"],
        }
    )
    with pytest.raises(_PANDERA_ERRORS):
        validate_sentiment_input(df)


def test_missing_sentence_fails():
    """Faltando coluna 'sentence' a validacao tem que falhar."""
    df = pd.DataFrame({"label": ["positive"]})
    with pytest.raises(_PANDERA_ERRORS):
        validate_sentiment_input(df)
