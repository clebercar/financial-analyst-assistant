"""Geracao de relatorio de drift com Evidently.

Compara dados de "treino" (referencia) com "producao" (sintetica por enquanto,
gerada com leve ruido sobre o proprio referencial). Quando tivermos dados reais
de producao basta apontar `current_csv` pra eles.

Outputs gerados em `evaluation/results/drift/`:
- `drift_report.html` -> relatorio interativo do Evidently
- `summary.json`      -> resumo machine-readable (drifted columns, share, p-values)

Notas sobre versoes:
- Evidently 0.7.x usa a API nova: `from evidently import Report, Dataset`
  (a antiga `evidently.report.Report` foi removida).
- O preset `DataDriftPreset` continua em `evidently.presets`.
- Em vez de `report.as_dict()` usamos `snapshot.dict()` (Snapshot e o objeto
  retornado por Report.run() na 0.7.x).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)


def _gerar_dados_sinteticos(
    reference: pd.DataFrame,
    n: int = 200,
    drift_factor: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dados de "producao" amostrando o referencial e adicionando ruido.

    O ruido proporcional ao desvio padrao de cada coluna garante que mesmo com
    `drift_factor` pequeno o teste estatistico (K-S) capture o drift — o que e
    desejavel pra demonstrar o relatorio.
    """
    rng = np.random.default_rng(seed)
    sample = reference.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    # Aplica ruido coluna a coluna escalado pelo std (preserva escala original).
    for col in sample.columns:
        std = float(sample[col].std()) or 1.0
        sample[col] = sample[col] + drift_factor * std * rng.normal(size=n)
    return sample


def _carregar_referencia(reference_csv: str) -> pd.DataFrame:
    """Carrega CSV de referencia; se ausente, baixa precos AAPL via yfinance."""
    p = Path(reference_csv)
    if p.exists():
        logger.info("Carregando referencia de %s", p)
        return pd.read_csv(p)

    logger.info("Referencia nao encontrada — gerando a partir de AAPL via yfinance")
    # Import local pra evitar custo se o caller passar um CSV pronto.
    from src.data.collector import baixar_dados_acao

    df = baixar_dados_acao("AAPL", "2023-01-01", "2024-12-31")
    ref = df[["Close", "Volume"]].copy().reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(p, index=False)
    return ref


def _extrair_summary(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
    """Extrai resumo machine-readable do snapshot do Evidently 0.7.x.

    O DataDriftPreset gera:
    - 1x DriftedColumnsCount com `value.count` e `value.share`
    - 1x ValueDrift por coluna com `value` = p-value (K-S)
    """
    metrics = snapshot_dict.get("metrics", [])
    drifted_count: float | None = None
    drifted_share: float | None = None
    column_pvalues: dict[str, float] = {}

    for m in metrics:
        name = m.get("metric_name", "")
        value = m.get("value")
        config = m.get("config", {}) or {}
        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drifted_count = value.get("count")
                drifted_share = value.get("share")
        elif name.startswith("ValueDrift"):
            col = config.get("column")
            if col is not None and isinstance(value, int | float):
                column_pvalues[str(col)] = float(value)

    return {
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "n_columns": len(column_pvalues),
        "column_pvalues": column_pvalues,
    }


def gerar_drift_report(
    reference_csv: str = "data/processed/lstm_features_reference.csv",
    output_dir: str = "evaluation/results/drift",
    current_csv: str | None = None,
) -> dict[str, Any]:
    """Roda Evidently sobre referencia x atual e salva HTML + summary JSON.

    Args:
        reference_csv: caminho do CSV de referencia (gerado se ausente).
        output_dir: diretorio de saida do relatorio.
        current_csv: caminho do CSV de "producao". Se None, gera dados
            sinteticos com leve drift sobre a referencia.

    Returns:
        dict com summary (drifted_columns_count, share, p-values por coluna).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = _carregar_referencia(reference_csv)

    if current_csv and Path(current_csv).exists():
        current = pd.read_csv(current_csv)
    else:
        current = _gerar_dados_sinteticos(ref)

    # Evidently 0.7.x espera Dataset (Dataset.from_pandas) ou DataFrame direto.
    # Usamos Dataset pra deixar explicito (e ter mais controle se quisermos
    # adicionar `data_definition` no futuro).
    ref_ds = Dataset.from_pandas(ref)
    cur_ds = Dataset.from_pandas(current)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    html_path = out / "drift_report.html"
    snapshot.save_html(str(html_path))

    summary = _extrair_summary(snapshot.dict())
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Drift summary: %s", summary)
    logger.info("HTML report salvo em %s", html_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    gerar_drift_report()
