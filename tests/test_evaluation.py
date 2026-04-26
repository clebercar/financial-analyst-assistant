"""Testes do pacote `evaluation/`.

Como o pipeline real consome a API do Gemini (custosa e dependente de chave),
todos os testes aqui mockam:
- O agente (`create_financial_agent`) -> retorna um stub com `.invoke`.
- `retrieve` do RAG -> retorna chunks estaticos.
- A funcao `evaluate` do RAGAS -> retorna scores fakes.
- O cliente Gemini (LLM-as-judge) -> retorna respostas com numeros fixos.

Objetivo: garantir que o fluxo "andamento -> persistencia" funciona sem rede.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures comuns
# ---------------------------------------------------------------------------


@pytest.fixture
def golden_set_tmp(tmp_path: Path) -> Path:
    """Cria um golden set minimo (3 itens) no diretorio temporario."""
    items = [
        {
            "id": "rag_test_01",
            "query": "O que diz o 10-K da Apple sobre cybersecurity?",
            "category": "rag_pure",
            "ticker": "AAPL",
            "tools_expected": ["buscar_em_filings"],
            "expected_answer": "A Apple cita riscos de cybersecurity no 10-K.",
            "expected_contexts": [],
        },
        {
            "id": "simple_test_01",
            "query": "Qual o preco da AAPL?",
            "category": "tool_simple",
            "ticker": "AAPL",
            "tools_expected": ["consultar_preco"],
            "expected_answer": "Preco atual via Yahoo Finance.",
            "expected_contexts": [],
        },
        {
            "id": "multi_test_01",
            "query": "Devo comprar AAPL?",
            "category": "multi_hop",
            "ticker": "AAPL",
            "tools_expected": ["buscar_em_filings", "consultar_preco"],
            "expected_answer": "Analise educacional, nao recomendacao financeira.",
            "expected_contexts": [],
        },
    ]
    p = tmp_path / "golden_set.json"
    p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Task 6.2 - RAGAS
# ---------------------------------------------------------------------------


def test_evaluate_pipeline_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, golden_set_tmp: Path
) -> None:
    """O pipeline RAGAS executa de ponta a ponta com mocks e persiste o JSON."""
    from evaluation import ragas_eval

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "output": "A Apple cita riscos cyber e responsabilidades regulatorias no 10-K.",
        "intermediate_steps": [],
    }
    monkeypatch.setattr(
        ragas_eval, "_load_golden", lambda p: json.loads(golden_set_tmp.read_text())
    )

    # Mocks dos imports tardios dentro de evaluate_pipeline.
    fake_create_agent = MagicMock(return_value=fake_agent)
    fake_get_coll = MagicMock(return_value="fake_collection")

    def fake_retrieve(query: str, coll: Any, embed_fn: Any, top_k: int = 3) -> list[dict]:
        return [
            {"trecho": "Cyber risks are mentioned in Item 1A.", "ticker": "AAPL"},
            {"trecho": "Apple invests in security infrastructure.", "ticker": "AAPL"},
        ]

    fake_embed = MagicMock(return_value=[0.0] * 768)

    fake_scores = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.92,
        "context_precision": 0.78,
        "context_recall": 0.81,
    }
    fake_evaluate = MagicMock(return_value=fake_scores)
    fake_dataset_cls = MagicMock()
    fake_dataset_cls.from_list = MagicMock(return_value="fake_dataset")

    # Substitui os modulos no sys.modules para os imports tardios pegarem o mock.
    # Os attributes sao setados dinamicamente; mypy nao consegue inferir, dai o
    # ignore.
    import sys
    import types

    fake_datasets_mod = types.ModuleType("datasets")
    fake_datasets_mod.Dataset = fake_dataset_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_mod)

    fake_ragas_mod = types.ModuleType("ragas")
    fake_ragas_mod.evaluate = fake_evaluate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas_mod)

    fake_metrics_mod = types.ModuleType("ragas.metrics")
    fake_metrics_mod.faithfulness = "faithfulness_metric"  # type: ignore[attr-defined]
    fake_metrics_mod.answer_relevancy = "answer_relevancy_metric"  # type: ignore[attr-defined]
    fake_metrics_mod.context_precision = "context_precision_metric"  # type: ignore[attr-defined]
    fake_metrics_mod.context_recall = "context_recall_metric"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics_mod)

    fake_rag_pipeline = types.ModuleType("src.agent.rag_pipeline")
    fake_rag_pipeline._gemini_embed = fake_embed  # type: ignore[attr-defined]
    fake_rag_pipeline.get_collection = fake_get_coll  # type: ignore[attr-defined]
    fake_rag_pipeline.retrieve = fake_retrieve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.agent.rag_pipeline", fake_rag_pipeline)

    fake_react = types.ModuleType("src.agent.react_agent")
    fake_react.create_financial_agent = fake_create_agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.agent.react_agent", fake_react)

    output_path = tmp_path / "ragas_scores.json"
    out = ragas_eval.evaluate_pipeline(
        golden_path=str(golden_set_tmp),
        output_path=str(output_path),
    )

    assert out["faithfulness"] == 0.85
    assert out["answer_relevancy"] == 0.92
    assert out["context_precision"] == 0.78
    assert out["context_recall"] == 0.81
    assert out["n_items"] == 3
    assert output_path.exists()
    persisted = json.loads(output_path.read_text())
    assert persisted == out
    fake_dataset_cls.from_list.assert_called_once()
    fake_evaluate.assert_called_once()


def test_build_rag_rows_handles_agent_exception() -> None:
    """Se o agente levantar excecao para uma pergunta, a row deve ter answer vazio."""
    from evaluation.ragas_eval import _build_rag_rows

    failing_agent = MagicMock()
    failing_agent.invoke.side_effect = RuntimeError("boom")

    def retrieve_fn(q: str) -> list[dict]:
        return [{"trecho": "ctx-1"}, {"trecho": "ctx-2"}]

    golden = [
        {"id": "x", "query": "Pergunta?", "expected_answer": "GT"},
    ]
    rows = _build_rag_rows(golden, failing_agent, retrieve_fn)
    assert len(rows) == 1
    assert rows[0]["answer"] == ""
    assert rows[0]["contexts"] == ["ctx-1", "ctx-2"]
    assert rows[0]["ground_truth"] == "GT"


# ---------------------------------------------------------------------------
# Task 6.3 - LLM-as-judge
# ---------------------------------------------------------------------------


def test_judge_response_parses_numeric_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """`judge_response` deve interpretar a resposta numerica do juiz para cada criterio."""
    from evaluation import llm_judge

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    fake_responses = iter(
        [
            MagicMock(text=" 4.5 "),
            MagicMock(text="3"),
            MagicMock(text="5.0"),
        ]
    )
    fake_model = MagicMock()
    fake_model.generate_content.side_effect = lambda _prompt: next(fake_responses)

    fake_genai = MagicMock()
    fake_genai.configure = MagicMock()
    fake_genai.GenerativeModel = MagicMock(return_value=fake_model)
    monkeypatch.setattr(llm_judge, "genai", fake_genai)

    criteria = {
        "coerencia_tecnica": "Avalie de 0 a 5.",
        "citacao_fontes": "Avalie de 0 a 5.",
        "completude": "Avalie de 0 a 5.",
    }
    scores = llm_judge.judge_response("Q?", "R.", criteria)
    assert scores == {
        "coerencia_tecnica": 4.5,
        "citacao_fontes": 3.0,
        "completude": 5.0,
    }
    fake_genai.configure.assert_called_once_with(api_key="fake-key")


def test_judge_response_lida_com_erro(monkeypatch: pytest.MonkeyPatch) -> None:
    """Se o juiz levantar erro num criterio, o score correspondente deve ser None."""
    from evaluation import llm_judge

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = RuntimeError("api boom")

    fake_genai = MagicMock()
    fake_genai.GenerativeModel = MagicMock(return_value=fake_model)
    monkeypatch.setattr(llm_judge, "genai", fake_genai)

    scores = llm_judge.judge_response("Q?", "R.", {"crit_a": "desc"})
    assert scores == {"crit_a": None}


def test_evaluate_with_judge_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, golden_set_tmp: Path
) -> None:
    """O pipeline LLM-as-judge persiste resultados por item com mocks."""
    from evaluation import llm_judge

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"output": "Resposta com fontes.", "intermediate_steps": []}

    import sys
    import types

    fake_react = types.ModuleType("src.agent.react_agent")
    fake_react.create_financial_agent = MagicMock(return_value=fake_agent)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.agent.react_agent", fake_react)

    monkeypatch.setattr(llm_judge, "judge_response", lambda q, a, c, **kw: {k: 4.0 for k in c})
    monkeypatch.setattr(
        llm_judge,
        "_load_criteria",
        lambda: {"coerencia_tecnica": "x", "citacao_fontes": "x", "completude": "x"},
    )

    output_path = tmp_path / "judge_scores.json"
    results = llm_judge.evaluate_with_judge(
        golden_path=str(golden_set_tmp),
        output_path=str(output_path),
    )
    assert len(results) == 3
    for r in results:
        assert set(r["scores"].keys()) == {"coerencia_tecnica", "citacao_fontes", "completude"}
        assert all(v == 4.0 for v in r["scores"].values())
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Task 6.4 - Benchmark
# ---------------------------------------------------------------------------


def test_run_benchmark_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, golden_set_tmp: Path
) -> None:
    """O benchmark deve rodar as 3 configs e medir latencia com agente mockado."""
    from evaluation import benchmark_configs

    invoke_calls: list[dict] = []

    def fake_invoke(payload: dict) -> dict:
        invoke_calls.append(payload)
        return {"output": f"resposta-{len(invoke_calls)}", "intermediate_steps": []}

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = fake_invoke

    creator = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(benchmark_configs, "_create_agent", creator)

    output_path = tmp_path / "benchmark.json"
    results = benchmark_configs.run_benchmark(
        golden_path=str(golden_set_tmp),
        output_path=str(output_path),
        n_items=2,
    )

    # 3 configs no benchmark.
    assert set(results.keys()) == {"A_baseline", "B_more_context", "C_smaller_model"}
    for payload in results.values():
        assert payload["n_sucesso"] == 2
        assert payload["latencia_media_s"] is not None
        assert len(payload["outputs"]) == 2
    assert output_path.exists()
    assert creator.call_count == 3


def test_run_benchmark_lida_com_falha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, golden_set_tmp: Path
) -> None:
    """Se uma config explodir em uma pergunta, o benchmark continua e marca o item como erro."""
    from evaluation import benchmark_configs

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = RuntimeError("api failure")
    monkeypatch.setattr(benchmark_configs, "_create_agent", lambda **_kw: fake_agent)

    output_path = tmp_path / "benchmark.json"
    results = benchmark_configs.run_benchmark(
        golden_path=str(golden_set_tmp),
        output_path=str(output_path),
        n_items=1,
    )
    for payload in results.values():
        assert payload["n_sucesso"] == 0
        assert payload["outputs"][0]["error"]
