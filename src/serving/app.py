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
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from src.monitoring.prometheus_metrics import (
    medir_tempo,
    metricas_response,
    registrar_erro,
    registrar_requisicao,
)
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
    """
    global _AGENT
    if _AGENT is None:
        # Import local (lazy) pra que o `from src.serving.app import app` nao
        # carregue langchain/langgraph se a API for usada so para /health.
        from src.agent.react_agent import create_financial_agent

        _AGENT = create_financial_agent()
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
    """Roda o agente ReAct sobre a pergunta e retorna resposta + tools usadas."""
    try:
        agent = get_agent()
    except RuntimeError as e:
        # GEMINI_API_KEY ausente ou config invalida.
        registrar_erro()
        raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        result = agent.invoke({"input": req.pergunta})
    except Exception as e:
        registrar_erro()
        logger.exception("Erro ao executar agente")
        raise HTTPException(status_code=500, detail=f"Erro no agente: {e}") from e

    intermediate = result.get("intermediate_steps", []) or []
    tools_used = [step[0].tool for step in intermediate]
    return ChatResponse(
        resposta=result.get("output", ""),
        iteracoes=len(intermediate),
        tools_chamadas=tools_used,
    )


@app.get("/metrics", tags=["Monitoramento"])
async def metricas():
    """Metricas no formato Prometheus (texto plano)."""
    return metricas_response()
