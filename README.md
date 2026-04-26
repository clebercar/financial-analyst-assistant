# Tech Challenge Fase 4 - Previsao de Precos com LSTM

Projeto da Fase 4 da pos-graduacao em Machine Learning Engineering (FIAP).

Modelo preditivo usando rede neural **LSTM (Long Short-Term Memory)** para prever o preco de fechamento de acoes da Apple (AAPL), com pipeline completa desde a coleta de dados ate o deploy em API REST containerizada.

## Estrutura do Projeto

```
phase-4/
├── notebooks/
│   └── exploratory_analysis.ipynb   # Analise exploratoria + treinamento
├── src/
│   ├── data/
│   │   └── collector.py             # Coleta de dados via Yahoo Finance
│   ├── model/
│   │   ├── lstm.py                  # Arquitetura do modelo LSTM
│   │   ├── preprocessing.py         # Normalizacao e criacao de sequencias
│   │   └── trainer.py               # Pipeline de treinamento e avaliacao
│   └── api/
│       ├── main.py                  # API FastAPI
│       ├── schemas.py               # Schemas de request/response
│       └── monitoring.py            # Metricas Prometheus
├── models/                          # Artefatos do modelo treinado
├── tests/                           # Testes automatizados
├── Dockerfile                       # Container da API
├── docker-compose.yml               # Orquestracao
└── requirements.txt                 # Dependencias Python
```

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Dados | yfinance, pandas, numpy |
| API | FastAPI + Uvicorn |
| Monitoramento | Prometheus (prometheus-client) |
| Container | Docker |
| Testes | pytest |

## Como Executar

### 1. Configurar o ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar o modelo

Opcao A - via notebook (recomendado pra visualizar graficos):
```bash
jupyter lab notebooks/exploratory_analysis.ipynb
```

Opcao B - via script:
```bash
python -m src.models.trainer
```

### 3. Rodar a API localmente

```bash
uvicorn src.serving.app:app --reload --port 8000
```

A documentacao interativa (Swagger) fica em: http://localhost:8000/docs

### 4. Rodar com Docker

```bash
docker-compose up --build
```

### 5. Rodar os testes

```bash
pytest tests/ -v
```

## Endpoints da API

| Metodo | Endpoint | Descricao |
|---|---|---|
| GET | `/health` | Verifica se a API e o modelo estao funcionando |
| POST | `/predict` | Recebe precos historicos e retorna previsao |
| GET | `/metrics` | Metricas no formato Prometheus |
| GET | `/docs` | Documentacao Swagger (automatica) |

### Exemplo de uso do `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"precos_fechamento": [150.0, 151.2, 149.8, ...]}'
```

Resposta:
```json
{
  "preco_previsto": 155.42,
  "simbolo": "AAPL",
  "modelo_versao": "1.0.0"
}
```

## Modelo LSTM

### Arquitetura
```
Input (60 dias, 1 feature)
  -> LSTM (50 neuronios) + Dropout (20%)
  -> LSTM (50 neuronios) + Dropout (20%)
  -> Dense (25 neuronios, ReLU)
  -> Dense (1 neuronio) -> Preco previsto
```

### Metricas de Avaliacao
- **MAE** (Mean Absolute Error): erro medio absoluto em dolares
- **RMSE** (Root Mean Square Error): penaliza erros grandes
- **MAPE** (Mean Absolute Percentage Error): erro medio percentual

## Monitoramento

A API expoe metricas no formato Prometheus em `/metrics`:
- `api_requisicoes_total`: total de requisicoes por endpoint e status
- `api_tempo_resposta_segundos`: histograma do tempo de resposta
- `modelo_previsoes_total`: total de previsoes realizadas
- `modelo_erros_total`: total de erros na inferencia
