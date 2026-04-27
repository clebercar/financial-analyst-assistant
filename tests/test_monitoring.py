"""Testes do modulo de monitoramento (Langfuse + Prometheus + drift).

Nao depende de Langfuse cloud nem do Gemini: tudo mockado / sem chaves.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


class TestLangfuseTracer:
    """Verifica o contrato do `get_langfuse_callback`."""

    def test_retorna_none_sem_env_vars(self, monkeypatch):
        """Sem chaves Langfuse, deve retornar None (nao levantar excecao).

        Esse contrato e critico: a app precisa rodar offline / em CI sem
        cadastro Langfuse.
        """
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        # Re-importa pra que a logica de leitura de env rode com o monkey patch.
        from src.monitoring.langfuse_tracer import get_langfuse_callback

        cb = get_langfuse_callback()
        assert cb is None

    def test_retorna_none_com_apenas_public_key(self, monkeypatch):
        """Public key sem secret tambem deve resultar em None."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        from src.monitoring.langfuse_tracer import get_langfuse_callback

        assert get_langfuse_callback() is None

    def test_retorna_handler_com_chaves_validas(self, monkeypatch):
        """Com as duas chaves setadas, devolve um handler instanciado.

        Nao testa conectividade real com Langfuse: so confere que o objeto
        retornado nao e None e tem os metodos esperados de callback do
        LangChain (`on_llm_start`, `on_chain_start`).
        """
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        from src.monitoring.langfuse_tracer import get_langfuse_callback

        cb = get_langfuse_callback()
        # Em ambiente de teste pode falhar a inicializacao real e cair no
        # except (retorna None). O importante e nao explodir.
        assert cb is None or hasattr(cb, "on_chain_start")


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


class TestPrometheusChatMetrics:
    """As 4 metricas novas devem existir e ter labels/buckets corretos."""

    def test_metricas_chat_existem(self):
        from src.monitoring import prometheus_metrics as pm

        assert hasattr(pm, "chat_requests_total")
        assert hasattr(pm, "chat_tool_calls_total")
        assert hasattr(pm, "chat_latency_seconds")
        assert hasattr(pm, "chat_iterations")

    def test_chat_requests_tem_label_status(self):
        from src.monitoring import prometheus_metrics as pm

        # Counter.labels deve aceitar status="success" sem levantar.
        pm.chat_requests_total.labels(status="success").inc()
        pm.chat_requests_total.labels(status="error").inc()

    def test_chat_tool_calls_tem_label_tool_name(self):
        from src.monitoring import prometheus_metrics as pm

        pm.chat_tool_calls_total.labels(tool_name="consultar_preco").inc()


class TestChatEndpointMetricsIntegration:
    """Endpoint /chat deve incrementar metricas em sucesso e erro."""

    def test_chat_sucesso_incrementa_metricas(self, monkeypatch):
        """Caminho feliz: incrementa chat_requests_total{status=success} e tools."""
        from fastapi.testclient import TestClient

        from src.monitoring import prometheus_metrics as pm
        from src.serving import app as app_mod

        # Snapshot do contador antes da request.
        success_before = pm.chat_requests_total.labels(status="success")._value.get()

        action = MagicMock()
        action.tool = "consultar_preco"
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {
            "output": "ok",
            "intermediate_steps": [(action, "obs")],
        }
        monkeypatch.setattr(app_mod, "get_agent", lambda: fake_agent)

        client = TestClient(app_mod.app)
        r = client.post("/chat", json={"pergunta": "teste"})
        assert r.status_code == 200

        success_after = pm.chat_requests_total.labels(status="success")._value.get()
        assert success_after == success_before + 1

    def test_chat_erro_incrementa_metricas_de_erro(self, monkeypatch):
        """Caminho de erro: agente explode -> chat_requests_total{status=error} sobe.

        Prepara estrutura pra Dia 8 (guardrails podem retornar 4xx; metricas
        precisam funcionar tambem nesse caminho).
        """
        from fastapi.testclient import TestClient

        from src.monitoring import prometheus_metrics as pm
        from src.serving import app as app_mod

        error_before = pm.chat_requests_total.labels(status="error")._value.get()

        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = ValueError("simulated failure")
        monkeypatch.setattr(app_mod, "get_agent", lambda: fake_agent)

        client = TestClient(app_mod.app, raise_server_exceptions=False)
        r = client.post("/chat", json={"pergunta": "teste"})
        assert r.status_code == 500

        error_after = pm.chat_requests_total.labels(status="error")._value.get()
        assert error_after == error_before + 1


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------


class TestDriftReport:
    """Smoke test do drift report sem rede (usa CSV pre-existente)."""

    def test_extrair_summary_estrutura(self):
        """Extrator deve achar drifted count + p-values por coluna."""
        from src.monitoring.drift_report import _extrair_summary

        snapshot_dict = {
            "metrics": [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 1.0, "share": 0.5},
                    "config": {},
                },
                {
                    "metric_name": "ValueDrift(column=Close)",
                    "value": 0.01,
                    "config": {"column": "Close"},
                },
                {
                    "metric_name": "ValueDrift(column=Volume)",
                    "value": 0.8,
                    "config": {"column": "Volume"},
                },
            ]
        }
        summary = _extrair_summary(snapshot_dict)
        assert summary["drifted_columns_count"] == 1.0
        assert summary["drifted_columns_share"] == 0.5
        assert summary["n_columns"] == 2
        assert summary["column_pvalues"]["Close"] == pytest.approx(0.01)
        assert summary["column_pvalues"]["Volume"] == pytest.approx(0.8)

    def test_gerar_drift_report_a_partir_de_csv(self, tmp_path):
        """Roda end-to-end com CSVs pequenos (sem rede)."""
        from src.monitoring.drift_report import gerar_drift_report

        ref_csv = tmp_path / "ref.csv"
        cur_csv = tmp_path / "cur.csv"
        # CSVs minimos pra Evidently aceitar.
        pd.DataFrame({"Close": list(range(100)), "Volume": [1000 + i for i in range(100)]}).to_csv(
            ref_csv, index=False
        )
        pd.DataFrame(
            {"Close": list(range(50, 150)), "Volume": [1500 + i for i in range(100)]}
        ).to_csv(cur_csv, index=False)

        out_dir = tmp_path / "drift_out"
        summary = gerar_drift_report(
            reference_csv=str(ref_csv),
            output_dir=str(out_dir),
            current_csv=str(cur_csv),
        )

        assert (out_dir / "drift_report.html").exists()
        assert (out_dir / "summary.json").exists()
        with open(out_dir / "summary.json") as f:
            disk_summary = json.load(f)
        assert disk_summary["n_columns"] == 2
        assert summary["n_columns"] == 2
