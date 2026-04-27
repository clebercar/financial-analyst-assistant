# Modulo de monitoramento da API.
# Usa Prometheus pra expor metricas que podem ser coletadas por ferramentas
# de monitoramento em producao (Grafana, por exemplo).
#
# As metricas que a gente rastreia:
# - Quantas requisicoes a API recebeu (total e por status)
# - Quanto tempo cada requisicao levou
# - Quantas previsoes foram feitas

import time
from functools import wraps

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Contador de requisicoes totais, separado por endpoint e status HTTP
requisicoes_total = Counter(
    "api_requisicoes_total",
    "Total de requisicoes recebidas pela API",
    ["metodo", "endpoint", "status"],
)

# Histograma do tempo de resposta (em segundos)
tempo_resposta = Histogram(
    "api_tempo_resposta_segundos",
    "Tempo de resposta das requisicoes em segundos",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Contador de previsoes realizadas
previsoes_total = Counter(
    "modelo_previsoes_total",
    "Total de previsoes realizadas pelo modelo",
)

# Contador de erros no modelo
erros_modelo = Counter(
    "modelo_erros_total",
    "Total de erros durante a inferencia do modelo",
)

# ---------------------------------------------------------------------------
# Metricas especificas do endpoint /chat (agente ReAct)
# ---------------------------------------------------------------------------
# Mantemos as metricas legadas acima (compatibilidade com dashboards antigos)
# e adicionamos abaixo as 4 metricas exigidas pelo Datathon Fase 5:
# requests, tools chamadas, latencia, iteracoes.

chat_requests_total = Counter(
    "chat_requests_total",
    "Total de requisicoes /chat por status (success/error)",
    ["status"],
)

chat_tool_calls_total = Counter(
    "chat_tool_calls_total",
    "Total de chamadas a cada tool do agente",
    ["tool_name"],
)

chat_latency_seconds = Histogram(
    "chat_latency_seconds",
    "Latencia do endpoint /chat em segundos",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

chat_iterations = Histogram(
    "chat_iterations",
    "Numero de iteracoes (intermediate_steps) do agente por request",
    buckets=(1, 2, 3, 5, 7, 10),
)


def registrar_requisicao(metodo: str, endpoint: str, status: int) -> None:
    requisicoes_total.labels(metodo=metodo, endpoint=endpoint, status=str(status)).inc()


def registrar_previsao() -> None:
    previsoes_total.inc()


def registrar_erro() -> None:
    erros_modelo.inc()


def medir_tempo(endpoint: str):
    """
    Decorator que mede o tempo de execucao de um endpoint.
    Registra no histograma do Prometheus.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            inicio = time.time()
            resultado = await func(*args, **kwargs)
            duracao = time.time() - inicio
            tempo_resposta.labels(endpoint=endpoint).observe(duracao)
            return resultado

        return wrapper

    return decorator


def metricas_response() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
