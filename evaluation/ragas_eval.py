"""Avaliacao do pipeline RAG+Agent com as 4 metricas RAGAS principais.

Metricas (todas no intervalo [0, 1], maior = melhor):
- `faithfulness`: a resposta tem suporte nos contextos recuperados? (combate alucinacao)
- `answer_relevancy`: a resposta e relevante para a pergunta?
- `context_precision`: os contextos recuperados sao realmente uteis para a resposta?
- `context_recall`: os contextos recuperados cobrem o ground truth?

Fluxo:
1. Carrega o golden set (`data/golden_set/golden_set.json`).
2. Para cada item: roda o agente, captura a resposta, recupera contextos via RAG.
3. Monta um `Dataset` HuggingFace e passa para `ragas.evaluate`.
4. Persiste em `evaluation/results/ragas_scores.json`.

Atencao: este modulo NAO deve ser executado se nao houver `GEMINI_API_KEY` definida
no `.env` (consome tokens). Para testes unitarios, use mocks (ver
`tests/test_evaluation.py`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_golden(golden_path: str) -> list[dict]:
    """Carrega lista de itens do golden set em JSON."""
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def _build_rag_rows(golden: list[dict], agent: Any, retrieve_fn: Any) -> list[dict]:
    """Para cada item do golden, gera linha {question, answer, contexts, ground_truth}.

    `agent` deve expor `.invoke({"input": query}) -> {"output": str, ...}`.
    `retrieve_fn` deve aceitar (query: str) e retornar list[dict] com chave 'trecho'.
    """
    rows: list[dict] = []
    for item in golden:
        q = item["query"]
        try:
            chunks = retrieve_fn(q)
            contexts = [c["trecho"] for c in chunks]
        except Exception as e:
            logger.warning("retrieve falhou em %s: %s", item.get("id"), e)
            contexts = []
        try:
            result = agent.invoke({"input": q})
            answer = result.get("output", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.warning("agent falhou em %s: %s", item.get("id"), e)
            answer = ""
        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.get("expected_answer", ""),
            }
        )
    return rows


def evaluate_pipeline(
    golden_path: str = "data/golden_set/golden_set.json",
    output_path: str = "evaluation/results/ragas_scores.json",
    top_k: int = 3,
) -> dict:
    """Roda RAGAS sobre o golden set e persiste os scores agregados.

    Args:
        golden_path: caminho do golden_set.json.
        output_path: onde salvar os scores agregados.
        top_k: quantos chunks recuperar do RAG por pergunta.

    Returns:
        dict com 4 metricas RAGAS (cada uma float em [0, 1]).
    """
    # Imports tardios: evita carregar ragas/agent quando o modulo e importado
    # apenas para inspecao (ex: pytest).
    import os

    from datasets import Dataset
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.agent.rag_pipeline import _gemini_embed, get_collection, retrieve
    from src.agent.react_agent import create_financial_agent

    golden = _load_golden(golden_path)
    agent = create_financial_agent()
    coll = get_collection()

    def retrieve_fn(q: str) -> list[dict]:
        return retrieve(q, coll, _gemini_embed, top_k=top_k)

    rows = _build_rag_rows(golden, agent, retrieve_fn)
    ds = Dataset.from_list(rows)

    # RAGAS por padrao usa OpenAI; passamos Gemini via LangChain wrappers
    # pra reutilizar a chave gratuita ja configurada.
    api_key = os.getenv("GEMINI_API_KEY")
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    )
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # RAGAS 0.4 retorna escala por amostra (list[float]). Calculamos a media,
    # ignorando NaN/None.
    import math

    def _aggregate(metric: str) -> float:
        raw = scores[metric]
        if isinstance(raw, int | float):
            return float(raw)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        clean = [v for v in raw if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(sum(clean) / len(clean)) if clean else 0.0

    def _per_sample(metric: str) -> list[float | None]:
        raw = scores[metric]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, list):
            return [
                None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                for v in raw
            ]
        return [float(raw)]

    out = {
        "faithfulness": _aggregate("faithfulness"),
        "answer_relevancy": _aggregate("answer_relevancy"),
        "context_precision": _aggregate("context_precision"),
        "context_recall": _aggregate("context_recall"),
        "n_items": len(rows),
        "per_sample": {
            "faithfulness": _per_sample("faithfulness"),
            "answer_relevancy": _per_sample("answer_relevancy"),
            "context_precision": _per_sample("context_precision"),
            "context_recall": _per_sample("context_recall"),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("RAGAS scores agregados: %s", {k: v for k, v in out.items() if k != "per_sample"})
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate_pipeline()
