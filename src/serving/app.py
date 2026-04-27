"""API REST do Datathon Fase 5: agente financeiro ReAct + Gemini.

Endpoints:
- GET  /health        -> liveness check.
- POST /chat          -> pergunta para o agente, retorna resposta + tools usadas.
- GET  /metrics       -> metricas Prometheus.

O agente e carregado lazy via `get_agent()` e cacheado em variavel modulo-global.
Isso evita inicializar o agente (que valida GEMINI_API_KEY) durante o import,
o que quebraria os testes que nao tem chave configurada.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from src.monitoring import prometheus_metrics as pm
from src.monitoring.prometheus_metrics import (
    medir_tempo,
    metricas_response,
    registrar_erro,
    registrar_requisicao,
)
from src.security.input_guardrail import validate_input
from src.security.output_guardrail import sanitize_output
from src.serving.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERSAO_API = "0.5.0"

app = FastAPI(
    title="Financial Analyst Agent API",
    description=(
        "API REST do Datathon Fase 5. Expoe um agente ReAct com 4 tools "
        "financeiras (consultar_preco, prever_preco_lstm, analisar_sentimento, "
        "buscar_em_filings) backed pelo Gemini."
    ),
    version=VERSAO_API,
)

# Cache do agente: criado na primeira chamada de `/chat` (lazy).
_AGENT: Any = None


def get_agent() -> Any:
    """Retorna o agente ReAct, criando-o na primeira chamada.

    Lazy load por dois motivos:
    1. Evita explodir no import quando GEMINI_API_KEY nao esta setada (caso dos
       testes unitarios).
    2. Evita pagar o custo de inicializacao do langgraph antes da primeira
       requisicao real chegar.

    Tambem tenta anexar o callback do Langfuse via env vars; se ausente,
    cria o agente sem tracing (comportamento totalmente opt-in).
    """
    global _AGENT
    if _AGENT is None:
        # Import local (lazy) pra que o `from src.serving.app import app` nao
        # carregue langchain/langgraph se a API for usada so para /health.
        from src.agent.react_agent import create_financial_agent
        from src.monitoring.langfuse_tracer import get_langfuse_callback

        cb = get_langfuse_callback()
        _AGENT = create_financial_agent(callbacks=[cb] if cb else None)
    return _AGENT


@app.middleware("http")
async def middleware_monitoramento(request: Request, call_next):
    """Registra cada requisicao no Prometheus."""
    response = await call_next(request)
    registrar_requisicao(
        metodo=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check() -> HealthResponse:
    """Liveness check simples - usado por orquestradores (k8s, docker-compose)."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["Agente"])
@medir_tempo(endpoint="/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Roda o agente ReAct sobre a pergunta e retorna resposta + tools usadas.

    Metricas Prometheus registradas (em try/except/finally pra cobrir sucesso E erro):
    - chat_requests_total{status=success|error}
    - chat_tool_calls_total{tool_name=<tool>}
    - chat_latency_seconds (histograma)
    - chat_iterations (histograma)
    """
    t0 = time.time()
    try:
        # Camada 1 (input guardrail): roda ANTES de qualquer chamada cara.
        # Bloqueio aqui evita pagar custo Gemini em prompt injection trivial.
        ok, motivo = validate_input(req.pergunta)
        if not ok:
            pm.chat_requests_total.labels(status="blocked_input").inc()
            logger.warning("Input bloqueado pelo guardrail: %s", motivo)
            raise HTTPException(status_code=400, detail=motivo)

        try:
            agent = get_agent()
        except RuntimeError as e:
            # GEMINI_API_KEY ausente ou config invalida.
            registrar_erro()
            pm.chat_requests_total.labels(status="error").inc()
            raise HTTPException(status_code=503, detail=str(e)) from e

        try:
            result = agent.invoke({"input": req.pergunta})
        except Exception as e:
            registrar_erro()
            pm.chat_requests_total.labels(status="error").inc()
            logger.exception("Erro ao executar agente")
            raise HTTPException(status_code=500, detail=f"Erro no agente: {e}") from e

        intermediate = result.get("intermediate_steps", []) or []
        tools_used = [step[0].tool for step in intermediate]

        # Sucesso: registra contadores e histogramas das tools/iteracoes.
        for tool_name in tools_used:
            pm.chat_tool_calls_total.labels(tool_name=tool_name).inc()
        pm.chat_iterations.observe(len(intermediate))
        pm.chat_requests_total.labels(status="success").inc()

        # Camada 2 (output guardrail): redaciona PII antes de devolver.
        resposta_bruta = result.get("output", "")
        resposta_sanitizada = sanitize_output(resposta_bruta)

        return ChatResponse(
            resposta=resposta_sanitizada,
            iteracoes=len(intermediate),
            tools_chamadas=tools_used,
        )
    finally:
        # Latencia ENTRA tanto em sucesso quanto em erro (HTTPException
        # propaga depois deste finally executar).
        pm.chat_latency_seconds.observe(time.time() - t0)


@app.get("/metrics", tags=["Monitoramento"])
async def metricas():
    """Metricas no formato Prometheus (texto plano)."""
    return metricas_response()
