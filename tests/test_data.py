# Testes do loader de dados do FinancialPhraseBank.
# A ideia aqui e validar a logica de carregamento e mapeamento de labels SEM
# baixar os ~600MB do dataset real. Mockamos o load_dataset do HuggingFace.

import pandas as pd

from src.data.financial_phrasebank import load_phrasebank


def test_load_returns_dataframe(monkeypatch):
    """Garante que load_phrasebank retorna DataFrame com colunas esperadas."""
    fake = pd.DataFrame(
        {
            "sentence": ["Apple posted strong earnings", "Stock fell 10%"],
            "label": ["positive", "negative"],
        }
    )

    def _fake_load(*args, **kwargs):
        class _DS:
            def to_pandas(self):
                return fake

        return {"train": _DS()}

    monkeypatch.setattr("src.data.financial_phrasebank.load_dataset", _fake_load)
    df = load_phrasebank()
    assert {"sentence", "label"}.issubset(df.columns)
    assert len(df) == 2


def test_load_maps_int_labels_to_strings(monkeypatch):
    """Quando o dataset retorna labels int (0/1/2), convertemos para strings."""
    fake = pd.DataFrame(
        {
            "sentence": ["bad news", "ok news", "good news"],
            "label": [0, 1, 2],
        }
    )

    def _fake_load(*args, **kwargs):
        class _DS:
            def to_pandas(self):
                return fake

        return {"train": _DS()}

    monkeypatch.setattr("src.data.financial_phrasebank.load_dataset", _fake_load)
    df = load_phrasebank()
    assert set(df["label"].tolist()) == {"negative", "neutral", "positive"}


def test_sec_edgar_filing_path_format(tmp_path):
    """build_filing_id deve gerar identificadores estaveis no formato TICKER_TIPO_ANO."""
    from src.data.sec_edgar import build_filing_id

    fid = build_filing_id("AAPL", "10-K", "2024")
    assert fid == "AAPL_10-K_2024"
