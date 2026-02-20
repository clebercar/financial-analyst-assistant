# Ponto de entrada da API REST.
# Aqui a gente configura o FastAPI, carrega o modelo treinado e define os endpoints.
# A API permite receber precos historicos e devolver a previsao do proximo dia.

import numpy as np
import joblib
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from tensorflow.keras.models import load_model

from src.api.schemas import PrevisaoRequest, PrevisaoResponse, HealthResponse
from src.api.monitoring import (
    registrar_requisicao,
    registrar_previsao,
    registrar_erro,
    medir_tempo,
    metricas_response,
)
from src.model.preprocessing import criar_scaler, normalizar_dados, desnormalizar_dados

# Configuracao de logging pra rastrear o que ta acontecendo
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Caminhos dos artefatos salvos
CAMINHO_MODELO = Path("models/lstm_model.keras")
CAMINHO_SCALER = Path("models/scaler.joblib")

# Configuracoes do modelo
SIMBOLO = "AAPL"
TAMANHO_JANELA = 60
VERSAO_MODELO = "1.0.0"

# Variaveis globais pro modelo e scaler (carregados uma vez na inicializacao)
modelo = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle da aplicacao: carrega o modelo quando a API sobe e limpa quando desce.
    Isso evita carregar o modelo a cada requisicao, o que seria muito lento.
    """
    global modelo, scaler

    logger.info("Iniciando API - carregando modelo e scaler...")

    if CAMINHO_MODELO.exists() and CAMINHO_SCALER.exists():
        modelo = load_model(CAMINHO_MODELO)
        scaler = joblib.load(CAMINHO_SCALER)
        logger.info("Modelo e scaler carregados com sucesso.")
    else:
        logger.warning(
            "Modelo ou scaler nao encontrados. A API vai subir, "
            "mas o endpoint de previsao nao vai funcionar ate treinar o modelo."
        )

    yield  # a API roda aqui

    logger.info("Encerrando API.")


app = FastAPI(
    title="LSTM Stock Price Predictor",
    description=(
        "API para previsao de precos de acoes usando modelo LSTM. "
        "Recebe precos historicos de fechamento e retorna a previsao do proximo dia."
    ),
    version=VERSAO_MODELO,
    lifespan=lifespan,
)


# Middleware pra registrar todas as requisicoes no monitoramento
@app.middleware("http")
async def middleware_monitoramento(request: Request, call_next):
    response = await call_next(request)
    registrar_requisicao(
        metodo=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """
    Verifica se a API ta funcionando e se o modelo ta carregado.
    Util pra monitoramento e pra saber se ta tudo de pe.
    """
    return HealthResponse(
        status="ok" if modelo is not None else "modelo_nao_carregado",
        modelo_carregado=modelo is not None,
        versao=VERSAO_MODELO,
    )


@app.post("/predict", response_model=PrevisaoResponse, tags=["Previsao"])
@medir_tempo(endpoint="/predict")
async def prever_preco(request: PrevisaoRequest):
    """
    Recebe uma lista de precos de fechamento historicos e retorna a previsao
    do preco do proximo dia.

    O modelo espera pelo menos 60 dias de dados. Se mandar mais, ele usa
    os ultimos 60 automaticamente.
    """
    if modelo is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo nao carregado. Treine o modelo antes de usar a API.",
        )

    try:
        # Pega os precos e garante que tem pelo menos a janela necessaria
        precos = np.array(request.precos_fechamento)

        if len(precos) < TAMANHO_JANELA:
            raise HTTPException(
                status_code=400,
                detail=f"Precisa de pelo menos {TAMANHO_JANELA} precos. Recebido: {len(precos)}",
            )

        # Usa os ultimos N dias (tamanho da janela)
        precos_janela = precos[-TAMANHO_JANELA:]

        # Normaliza usando o mesmo scaler do treinamento
        precos_normalizados = scaler.transform(precos_janela.reshape(-1, 1))

        # Formata pro LSTM: (1 amostra, 60 timesteps, 1 feature)
        entrada = precos_normalizados.reshape(1, TAMANHO_JANELA, 1)

        # Faz a previsao
        previsao_normalizada = modelo.predict(entrada, verbose=0)

        # Volta pra escala original (dolares)
        preco_previsto = float(scaler.inverse_transform(previsao_normalizada)[0, 0])

        registrar_previsao()

        logger.info(f"Previsao realizada: ${preco_previsto:.2f}")

        return PrevisaoResponse(
            preco_previsto=round(preco_previsto, 2),
            simbolo=SIMBOLO,
            modelo_versao=VERSAO_MODELO,
        )

    except HTTPException:
        raise
    except Exception as e:
        registrar_erro()
        logger.error(f"Erro na previsao: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno na previsao: {str(e)}")


@app.get("/metrics", tags=["Monitoramento"])
async def metricas():
    """
    Retorna as metricas no formato Prometheus.
    Pode ser consumido por ferramentas como Grafana pra criar dashboards.
    """
    return metricas_response()
