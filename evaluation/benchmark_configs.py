"""Benchmark comparando 3 configuracoes do agente.

Configuracoes comparadas:
- A_baseline       : modelo gemini-2.5-flash, top_k=3.
- B_more_context   : mesmo modelo, top_k=5 (mais contexto no RAG).
- C_smaller_model  : modelo gemini-2.5-flash-lite, top_k=3 (modelo menor).

Para cada configuracao rodamos um subset do golden set, medindo:
- latencia media por pergunta (segundos)
- numero de respostas bem sucedidas
- saidas brutas para inspecao posterior

Resultado em `evaluation/results/benchmark.json`.

Atencao: este modulo NAO deve ser executado se nao houver `GEMINI_API_KEY` no
`.env`. Para testes, use mocks (ver `tests/test_evaluation.py`).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


CONFIGS: list[dict[str, Any]] = [
    {
        "id": "A_baseline",
        "model": "gemini-2.5-flash",
        "top_k": 3,
        "desc": "Baseline: Gemini 2.5 Flash + top_k=3",
    },
    {
        "id": "B_more_context",
        "model": "gemini-2.5-flash",
        "top_k": 5,
        "desc": "Mais contexto: top_k=5",
    },
    {
        "id": "C_smaller_model",
        "model": "gemini-2.5-flash-lite",
        "top_k": 3,
        "desc": "Modelo menor: Gemini 2.5 Flash Lite",
    },
]


def _load_golden(golden_path: str) -> list[dict]:
    """Carrega o golden set."""
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def _create_agent(model: str) -> Any:
    """Cria um agente com o modelo especificado.

    Wrapper fino sobre `create_financial_agent` para facilitar mock em testes.
    """
    from src.agent.react_agent import create_financial_agent

    return create_financial_agent(model_name=model)


def _run_one_config(
    cfg: dict[str, Any],
    golden_subset: list[dict],
) -> dict[str, Any]:
    """Roda uma config sobre `golden_subset` e devolve o resumo."""
    logger.info("Config %s (%s)", cfg["id"], cfg["desc"])
    try:
        agent = _create_agent(model=cfg["model"])
    except Exception as e:
        logger.warning("Falha criando agente para %s: %s", cfg["id"], e)
        return {
            "config": cfg,
            "latencia_media_s": None,
            "n_sucesso": 0,
            "outputs": [{"id": item["id"], "error": str(e)} for item in golden_subset],
        }

    latencias: list[float] = []
    outputs: list[dict[str, Any]] = []
    for item in golden_subset:
        t0 = time.time()
        try:
            r = agent.invoke({"input": item["query"]})
            latencias.append(time.time() - t0)
            answer = r.get("output", "") if isinstance(r, dict) else str(r)
            outputs.append({"id": item["id"], "answer": answer})
        except Exception as e:
            logger.warning("%s falhou em %s: %s", cfg["id"], item.get("id"), e)
            outputs.append({"id": item["id"], "error": str(e)})

    media = sum(latencias) / len(latencias) if latencias else None
    return {
        "config": cfg,
        "latencia_media_s": media,
        "n_sucesso": len(latencias),
        "outputs": outputs,
    }


def run_benchmark(
    golden_path: str = "data/golden_set/golden_set.json",
    output_path: str = "evaluation/results/benchmark.json",
    n_items: int = 5,
) -> dict[str, Any]:
    """Roda as 3 configs sobre os primeiros `n_items` do golden set.

    Args:
        golden_path: caminho do golden_set.json.
        output_path: onde salvar os resultados.
        n_items: quantos itens do golden usar por config (subset rapido).

    Returns:
        dict {config_id: resultado_da_config}.
    """
    golden = _load_golden(golden_path)
    subset = golden[:n_items]

    results: dict[str, Any] = {}
    for cfg in CONFIGS:
        results[cfg["id"]] = _run_one_config(cfg, subset)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark salvo em %s", output_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_benchmark()
