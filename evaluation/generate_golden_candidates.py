"""Gera candidatos de pares (pergunta, resposta esperada) para o golden set.

O agente financeiro tem 4 categorias de comportamento que precisamos avaliar:

1. `rag_pure`     - perguntas que SO o RAG resolve (informacao de filings).
2. `tool_simple`  - perguntas que dependem de uma tool simples (preco, sentimento).
3. `tool_lstm`    - perguntas que envolvem a previsao LSTM.
4. `multi_hop`    - perguntas que combinam 2+ tools (RAG + preco, p.ex.).

Saida: `data/golden_set/candidates.json` com 30 candidatos. O autor edita
manualmente os campos `expected_answer` e `expected_contexts`, e seleciona
20 entradas finais em `data/golden_set/golden_set.json`.

Uso:
    python evaluation/generate_golden_candidates.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Templates de perguntas por categoria. Cada item: (pergunta, ticker_alvo|None).
# Meta: 30 candidatos no total (12 rag, 6 simple, 5 lstm, 7 multi_hop) para
# o autor curar 20 finais no golden_set.json.
RAG_PURE: list[tuple[str, str | None]] = [
    ("Quais os principais fatores de risco mencionados no ultimo 10-K da Apple?", "AAPL"),
    ("Como a Microsoft descreve sua estrategia de cloud no ultimo 10-K?", "MSFT"),
    (
        "Quais segmentos de receita do Google tem maior margem segundo o ultimo 10-K?",
        "GOOGL",
    ),
    ("O que a NVIDIA diz sobre concentracao de clientes no 10-K?", "NVDA"),
    ("Quais os principais investimentos da Meta em IA no ultimo 10-K?", "META"),
    ("Como a Apple aborda riscos cambiais no 10-K?", "AAPL"),
    ("Quais as obrigacoes legais pendentes da Microsoft no ultimo 10-Q?", "MSFT"),
    ("Que compromissos ESG o Google divulga no 10-K?", "GOOGL"),
    ("Quais sao os principais competidores citados pela NVIDIA no 10-K?", "NVDA"),
    ("Como a Meta descreve riscos regulatorios em seu ultimo 10-K?", "META"),
    ("Quais sao as fontes de receita do iPhone segundo o ultimo 10-K da Apple?", "AAPL"),
    ("Que metricas de engajamento a Microsoft destaca no ultimo 10-K?", "MSFT"),
]

TOOL_SIMPLE: list[tuple[str, str | None]] = [
    ("Qual o preco atual da NVDA?", "NVDA"),
    ("Como a META se comportou nos ultimos 30 dias?", "META"),
    ("Qual o volume medio de negociacao da AAPL?", "AAPL"),
    (
        "O sentimento do trecho 'Apple posted record revenue this quarter' e positivo?",
        None,
    ),
    ("Qual a variacao de preco da MSFT nos ultimos 30 dias?", "MSFT"),
    (
        "Classifique o sentimento de 'The company reported a significant decline in earnings'.",
        None,
    ),
]

TOOL_LSTM: list[tuple[str, str | None]] = [
    ("Qual sua previsao de preco da AAPL para os proximos 5 dias?", "AAPL"),
    ("Projete o preco da AAPL para 3 dias uteis.", "AAPL"),
    ("Tente prever a AAPL para os proximos 7 dias.", "AAPL"),
    ("Faca uma previsao de preco para a NVDA usando o modelo LSTM (5 dias).", "NVDA"),
    ("Quais sao as projecoes do modelo LSTM para AAPL nos proximos 10 dias?", "AAPL"),
]

MULTI_HOP: list[tuple[str, str | None]] = [
    (
        "Considerando o ultimo 10-K, o preco atual e a projecao LSTM, " "devo comprar AAPL?",
        "AAPL",
    ),
    (
        "A Microsoft esta bem posicionada considerando o 10-K e o preco recente?",
        "MSFT",
    ),
    ("Compare os riscos mencionados no 10-K da Apple e da NVIDIA.", None),
    ("O sentimento do mercado sobre a META justifica o preco atual?", "META"),
    (
        "Dado o 10-K e as previsoes, quais riscos eu corro investindo na AAPL?",
        "AAPL",
    ),
    (
        "Avalie o NVDA combinando os fatores de risco do 10-K com a "
        "performance de preco recente.",
        "NVDA",
    ),
    (
        "Resuma a tese de investimento da Apple usando o ultimo 10-K, o preco "
        "atual e a projecao LSTM.",
        "AAPL",
    ),
]


def _make_entry(
    item_id: str,
    query: str,
    category: str,
    ticker: str | None,
    tools_expected: list[str],
) -> dict:
    """Monta um candidato com placeholders para revisao manual."""
    return {
        "id": item_id,
        "query": query,
        "category": category,
        "ticker": ticker,
        "tools_expected": tools_expected,
        "expected_answer": "<EDITAR: resposta esperada com base no filing/dado>",
        "expected_contexts": [],
    }


def gerar_candidatos() -> list[dict]:
    """Gera a lista de candidatos sem persistir. Util para testes."""
    candidates: list[dict] = []

    for i, (q, ticker) in enumerate(RAG_PURE, 1):
        candidates.append(
            _make_entry(
                item_id=f"rag_{i:02d}",
                query=q,
                category="rag_pure",
                ticker=ticker,
                tools_expected=["buscar_em_filings"],
            )
        )

    for i, (q, ticker) in enumerate(TOOL_SIMPLE, 1):
        # ticker None -> sentiment-only
        tool = "consultar_preco" if ticker else "analisar_sentimento"
        candidates.append(
            _make_entry(
                item_id=f"simple_{i:02d}",
                query=q,
                category="tool_simple",
                ticker=ticker,
                tools_expected=[tool],
            )
        )

    for i, (q, ticker) in enumerate(TOOL_LSTM, 1):
        candidates.append(
            _make_entry(
                item_id=f"lstm_{i:02d}",
                query=q,
                category="tool_lstm",
                ticker=ticker,
                tools_expected=["prever_preco_lstm"],
            )
        )

    for i, (q, ticker) in enumerate(MULTI_HOP, 1):
        candidates.append(
            _make_entry(
                item_id=f"multi_{i:02d}",
                query=q,
                category="multi_hop",
                ticker=ticker,
                tools_expected=["buscar_em_filings", "consultar_preco"],
            )
        )

    return candidates


def main(output_path: str = "data/golden_set/candidates.json") -> None:
    """Gera candidatos e persiste em `output_path`."""
    candidates = gerar_candidatos()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    logger.info("Gerados %d candidatos em %s", len(candidates), out)
    print(f"Gerados {len(candidates)} candidatos em {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
