# Tech Challenge - Fase 5: Datathon (Assistente de Analista Financeiro)

## Estado atual (Fase 5 — Datathon)

Este projeto evoluiu da Fase 4 (LSTM AAPL) pra implementar o Datathon da
Fase 5 (LLMs + Agentes). Veja:
- Spec: `docs/superpowers/specs/2026-04-26-datathon-fase05-design.md`
- Plano: `docs/superpowers/plans/2026-04-26-datathon-fase05-implementacao.md`

Domínio: assistente de analista financeiro.
LLM: Gemini 2.0 Flash via API.

A documentação abaixo é o guia da Fase 4 (preservado por contexto histórico
do código que sobreviveu — preprocessing/yfinance/scaler — e da arquitetura LSTM).

---

# Tech Challenge - Fase 4: LSTM para Previsao de Acoes (legado)

## Visao Geral do Projeto

Criar um modelo preditivo **LSTM (Long Short-Term Memory)** para prever o valor de
fechamento de acoes da bolsa de valores, com pipeline completa: desde coleta de dados
ate deploy em API REST com Docker.

---

## Requisitos do Desafio (PDF)

| # | Requisito | Descricao |
|---|-----------|-----------|
| 1 | Coleta e Pre-processamento | Dados historicos via `yfinance`, normalizacao, split treino/teste |
| 2 | Modelo LSTM | Construcao, treinamento, avaliacao (MAE, RMSE, MAPE) |
| 3 | Salvamento do Modelo | Exportar modelo treinado para inferencia |
| 4 | Deploy (API) | API RESTful com FastAPI para previsoes |
| 5 | Monitoramento | Metricas de performance em producao |

### Entregaveis
- [ ] Codigo-fonte do modelo LSTM no repositorio Git + documentacao
- [ ] Scripts ou conteineres Docker para deploy da API
- [ ] Link para a API em producao (se deployada em nuvem)
- [ ] Video mostrando e explicando o funcionamento da API

---

## Decisoes Tecnicas

| Aspecto | Escolha | Justificativa |
|---------|---------|---------------|
| Framework ML | TensorFlow/Keras | Sintaxe mais simples para LSTM, ideal para aprendizado |
| API | FastAPI | Moderno, async, documentacao automatica (Swagger) |
| Container | Docker | Requisito do desafio |
| Monitoramento | Prometheus + logging estruturado | Leve, padrao da industria |
| Acao escolhida | AAPL (Apple) | Empresa com altissima liquidez, muito usada em tutoriais de ML |
| Python | 3.11+ | Compatibilidade com TensorFlow |

---

## Estrutura do Projeto

```
phase-4/
├── CLAUDE.md                    # Este arquivo (guia do projeto)
├── README.md                    # Documentacao para entrega
├── pos_tech_mlet_tech_challenge_fase_4.pdf
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Analise exploratoria + treinamento do modelo
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── collector.py         # Coleta de dados via yfinance
│   ├── model/
│   │   ├── __init__.py
│   │   ├── lstm.py              # Definicao do modelo LSTM
│   │   ├── preprocessing.py     # Normalizacao e criacao de sequencias
│   │   └── trainer.py           # Treinamento e avaliacao
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app principal
│       ├── schemas.py           # Pydantic schemas (request/response)
│       └── monitoring.py        # Metricas e monitoramento
│
├── models/                      # Modelos treinados salvos (.keras)
│   └── .gitkeep
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_api.py
```

---

## Conceitos-Chave para Apresentacao

### O que e LSTM?
LSTM (Long Short-Term Memory) e um tipo de rede neural recorrente (RNN) projetada
para aprender dependencias de longo prazo em sequencias de dados. Diferente de RNNs
tradicionais, LSTMs possuem um mecanismo de "portas" (gates) que controla o fluxo
de informacao:

- **Forget Gate**: decide quais informacoes descartar do estado anterior
- **Input Gate**: decide quais novas informacoes armazenar
- **Output Gate**: decide o que enviar como saida

Isso resolve o problema de "vanishing gradient" das RNNs tradicionais, permitindo
que a rede "lembre" padroes de longo prazo nos precos das acoes.

### Por que LSTM para acoes?
Precos de acoes sao **series temporais** - dados sequenciais onde a ordem importa.
LSTMs sao excelentes para capturar padroes como tendencias, sazonalidade e ciclos
que existem nesses dados.

### Pipeline do Projeto (fluxo de dados)
```
Yahoo Finance (yfinance)
    |
    v
Dados Brutos (Open, High, Low, Close, Volume)
    |
    v
Pre-processamento (MinMaxScaler normaliza para [0,1])
    |
    v
Criacao de Sequencias (janela deslizante de N dias)
    |
    v
Split Treino/Teste (80%/20%)
    |
    v
Modelo LSTM (camadas LSTM + Dense)
    |
    v
Treinamento (minimizar erro entre previsao e real)
    |
    v
Avaliacao (MAE, RMSE, MAPE)
    |
    v
Salvamento (.keras)
    |
    v
API FastAPI (carrega modelo, recebe dados, retorna previsao)
    |
    v
Docker (containeriza tudo para deploy)
```

### Metricas de Avaliacao
- **MAE (Mean Absolute Error)**: media dos erros absolutos. "Em media, erramos R$X"
- **RMSE (Root Mean Square Error)**: penaliza erros grandes. "Erros grandes pesam mais"
- **MAPE (Mean Absolute Percentage Error)**: erro em %. "Erramos X% em media"

---

## Guia de Implementacao Passo a Passo

### Passo 1: Setup do Ambiente
```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

**Dependencias principais:**
- `tensorflow` - framework de deep learning (contem Keras)
- `yfinance` - coleta de dados de acoes
- `scikit-learn` - MinMaxScaler para normalizacao
- `pandas` / `numpy` - manipulacao de dados
- `matplotlib` - graficos
- `fastapi` / `uvicorn` - API REST
- `prometheus-client` - monitoramento
- `joblib` - salvar o scaler
- `pytest` / `httpx` - testes

### Passo 2: Coleta de Dados (`src/data/collector.py`)
- Usar `yfinance` para baixar dados historicos de AAPL (Apple)
- Periodo sugerido: 2018-01-01 a 2024-12-31
- Colunas de interesse: principalmente `Close` (fechamento)
- Tratar dados faltantes (remover ou interpolar)

### Passo 3: Pre-processamento (`src/model/preprocessing.py`)
- **Normalizacao**: usar `MinMaxScaler` para escalar valores para [0, 1]
  (redes neurais funcionam melhor com dados normalizados)
- **Criacao de sequencias**: janela deslizante (ex: usar 60 dias para prever o proximo)
  - Input: [dia1, dia2, ..., dia60]
  - Output: dia61
- **Split**: 80% treino, 20% teste (sem embaralhar - e serie temporal!)

### Passo 4: Modelo LSTM (`src/model/lstm.py`)
Arquitetura sugerida:
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```
- **LSTM(50)**: 50 neuronios na camada LSTM
- **return_sequences=True**: primeira LSTM passa sequencia completa para a proxima
- **Dropout(0.2)**: desliga 20% dos neuronios aleatoriamente (evita overfitting)
- **Dense(25)**: camada densa intermediaria
- **Dense(1)**: saida unica (preco previsto)

### Passo 5: Treinamento (`src/model/trainer.py`)
- Optimizer: `adam` (adapta taxa de aprendizado automaticamente)
- Loss: `mean_squared_error` (padrao para regressao)
- Epochs: 50-100 (monitorar se o loss para de diminuir)
- Batch size: 32
- Salvar historico de treinamento para visualizacao

### Passo 6: Avaliacao
- Calcular MAE, RMSE e MAPE no conjunto de teste
- Plotar grafico: precos reais vs previstos
- Verificar se o modelo captura a tendencia geral

### Passo 7: Salvamento do Modelo
- Salvar modelo: `model.save('models/lstm_model.keras')`
- Salvar scaler: `joblib.dump(scaler, 'models/scaler.joblib')`
- Ambos sao necessarios para inferencia (o scaler inverte a normalizacao)

### Passo 8: API FastAPI (`src/api/main.py`)
Endpoints:
- `GET /health` - Health check
- `POST /predict` - Recebe dados historicos, retorna previsao
- `GET /metrics` - Metricas Prometheus
- `GET /docs` - Documentacao Swagger (automatica do FastAPI)

### Passo 9: Docker
- Dockerfile multi-stage (build + runtime)
- docker-compose.yml para facilitar execucao
- Expor porta 8000

### Passo 10: Monitoramento (`src/api/monitoring.py`)
- Tempo de resposta das requisicoes
- Contagem de requisicoes (total, sucesso, erro)
- Metricas expostas em `/metrics` (formato Prometheus)

---

## Comandos Uteis

```bash
# Treinar o modelo (via notebook ou script)
python -m src.model.trainer

# Rodar a API localmente
uvicorn src.api.main:app --reload --port 8000

# Rodar com Docker
docker build -t lstm-stock-api .
docker run -p 8000:8000 lstm-stock-api

# Rodar com docker-compose
docker-compose up --build

# Rodar testes
pytest tests/ -v

# Acessar documentacao da API
# Abrir no navegador: http://localhost:8000/docs
```

---

## Referencia do Repositorio FIAP

O repositorio https://github.com/FIAP/Pos_Tech_MLET/tree/deep-learning contem:
- `src/architectures/rnns/lstm.py` - Implementacao LSTM em PyTorch (referencia)
- `productization/` - Exemplo de deploy com FastAPI + Docker + Terraform
- `productization/src/app/model/lstm.py` - Modelo LSTM para producao
- `productization/Dockerfile` - Container com CUDA/GPU

**Nota**: O repo de referencia usa PyTorch. Nosso projeto usa TensorFlow/Keras por
ser mais acessivel para aprendizado, mas os conceitos sao identicos.

---

## Dicas para a Apresentacao (Video)

1. **Comece pelo problema**: "Queremos prever precos de acoes usando Deep Learning"
2. **Explique LSTM brevemente**: as 3 portas e por que e melhor que RNN simples
3. **Mostre o notebook**: graficos de dados, treinamento, metricas
4. **Demo da API**: abrir Swagger, enviar request, mostrar resposta
5. **Docker**: mostrar o container rodando
6. **Metricas**: mostrar endpoint `/metrics`
7. **Conclusao**: limitacoes (mercado e imprevisivel) e proximos passos

---

## Convencoes de Codigo

- Codigo e comentarios em **portugues** (para alinhamento com a apresentacao)
- Docstrings explicativas em cada funcao (voce precisa entender para apresentar)
- Type hints em todas as funcoes
- Nomes de variaveis descritivos
- Cada arquivo deve ter um comentario no topo explicando seu proposito
