"""LLM-as-judge: avalia respostas do agente em 3 criterios usando o Gemini.

Os criterios sao definidos em `configs/prompts.yaml` (chave `judge_criteria`):
- `coerencia_tecnica`  - usa terminologia financeira corretamente?
- `citacao_fontes`     - cita filing/data/ticker? (KPI DE NEGOCIO)
- `completude`         - aborda todos os aspectos da pergunta?

Cada criterio e avaliado em escala 0-5 (decimal aceito). O juiz e instruido a
responder APENAS com um numero. Se a chamada falhar, o score correspondente
e None.

Atencao: este modulo NAO deve ser executado se nao houver `GEMINI_API_KEY` no
`.env`. Para testes, use mocks (ver `tests/test_evaluation.py`).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import google.generativeai as genai  # type: ignore[import-not-found]
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _load_criteria(prompts_path: str = "configs/prompts.yaml") -> dict[str, str]:
    """Carrega os criterios `judge_criteria` do prompts.yaml."""
    with open(prompts_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["judge_criteria"]


def _load_golden(golden_path: str) -> list[dict]:
    """Carrega o golden set."""
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def judge_response(
    question: str,
    answer: str,
    criteria: dict[str, str],
    model: str = "gemini-2.0-flash",
) -> dict[str, float | None]:
    """Avalia uma unica (pergunta, resposta) em multiplos criterios.

    Args:
        question: a pergunta original feita ao agente.
        answer: a resposta produzida pelo agente.
        criteria: dict {nome_criterio: descricao} (vindo de prompts.yaml).
        model: modelo Gemini a usar como juiz.

    Returns:
        dict {nome_criterio: float [0,5] | None} - None se a chamada falhou.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao definida no .env")
    genai.configure(api_key=api_key)
    judge = genai.GenerativeModel(model)

    scores: dict[str, float | None] = {}
    for crit_name, crit_desc in criteria.items():
        prompt = (
            "Voce e um juiz avaliando respostas de um agente financeiro.\n\n"
            f"PERGUNTA: {question}\n"
            f"RESPOSTA: {answer}\n\n"
            f"CRITERIO ({crit_name}):\n{crit_desc}\n\n"
            "Responda APENAS com um numero de 0 a 5 (pode ser decimal, ex 3.5).\n"
            "Nao escreva nada alem do numero.\n"
        )
        try:
            r = judge.generate_content(prompt)
            scores[crit_name] = float(r.text.strip())
        except Exception as e:
            logger.warning("Judge falhou em %s: %s", crit_name, e)
            scores[crit_name] = None
    return scores


def evaluate_with_judge(
    golden_path: str = "data/golden_set/golden_set.json",
    output_path: str = "evaluation/results/judge_scores.json",
    model: str = "gemini-2.0-flash",
) -> list[dict[str, Any]]:
    """Roda o agente sobre o golden set e avalia cada resposta com o juiz LLM.

    Args:
        golden_path: caminho do golden_set.json.
        output_path: onde salvar a lista de resultados.
        model: modelo Gemini a usar como juiz.

    Returns:
        Lista de dicts {id, scores: {criterio: float|None}}.
    """
    # Import tardio: nao queremos custar uma chamada Gemini se este modulo
    # for so importado em testes.
    from src.agent.react_agent import create_financial_agent

    golden = _load_golden(golden_path)
    criteria = _load_criteria()
    agent = create_financial_agent()

    results: list[dict[str, Any]] = []
    for item in golden:
        try:
            r = agent.invoke({"input": item["query"]})
            answer = r.get("output", "") if isinstance(r, dict) else str(r)
        except Exception as e:
            logger.warning("Agente falhou em %s: %s", item.get("id"), e)
            answer = ""
        scores = judge_response(item["query"], answer, criteria, model=model)
        results.append({"id": item["id"], "scores": scores})
        logger.info("%s: %s", item["id"], scores)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate_with_judge()
