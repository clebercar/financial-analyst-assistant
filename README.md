# Datathon Fase 05 — Assistente de Analista Financeiro

[![tests](https://img.shields.io/badge/tests-72%20passing-brightgreen)]()
[![coverage](https://img.shields.io/badge/coverage-%E2%89%A560%25-blue)]()
[![python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

Sistema MLOps end-to-end que ajuda analistas de buy-side a decidir compra/venda
de acoes via agente conversacional com **RAG sobre filings 10-K/10-Q da SEC**,
**LSTM em PyTorch**, **classificador de sentimento sklearn** e **agente ReAct com
Gemini 2.0 Flash** — tudo com observabilidade, guardrails de seguranca e
governanca documentada.

> **Pitch (30s):** analistas gastam horas lendo um unico 10-K. Aqui um agente
> ReAct combina o filing mais recente da SEC, o preco de mercado, a projecao do
> LSTM e a leitura de sentimento — e devolve um sumario citavel em segundos,
> com trace completo em Langfuse, metricas em Grafana e PII filtrado por
> Presidio. Construido em 9 dias, integrando ~30% de reuso da Fase 4 (LSTM AAPL).

---

## Stack

| Camada           | Tecnologia                                                        |
|------------------|-------------------------------------------------------------------|
| Deep Learning    | PyTorch 2.x (LSTM)                                                |
| ML classico      | scikit-learn (TF-IDF + Logistic Regression de sentimento)         |
| Agente / LLM     | LangChain ReAct + Gemini 2.0 Flash                                |
| RAG              | ChromaDB + Gemini embeddings + chunking de filings SEC EDGAR      |
| API              | FastAPI + Uvicorn                                                 |
| Tracking         | MLflow (experimentos + model registry)                            |
| Tracing LLM      | Langfuse (traces do agente, custos, latencia por step)            |
| Metricas         | Prometheus + Grafana                                              |
| Drift            | Evidently (PSI offline)                                           |
| Seguranca        | Regex anti-injection + Microsoft Presidio (PII redaction)         |
| Container        | Docker + docker-compose                                           |
| Qualidade        | pytest, ruff, mypy, bandit, pandera                               |

---

## Demo rapida

### Pre-requisitos

- Python 3.11+
- Docker + docker-compose (recomendado)
- Chave da Gemini API ([Google AI Studio](https://aistudio.google.com/app/apikey))
- (Opcional) Conta Langfuse free tier ([cloud.langfuse.com](https://cloud.langfuse.com))

### Setup

```bash
git clone <repo>
cd phase-5

cp .env.example .env
# editar .env e colocar GEMINI_API_KEY (e Langfuse keys, se quiser tracing)

make install
```

### Pipeline completo (treino -> indexacao -> serve)

```bash
make train-lstm          # treina LSTM AAPL e loga no MLflow
make train-sentiment     # treina classificador de sentimento
make download-filings    # baixa 10-K/10-Q da SEC (precisa SEC_USER_AGENT no .env)
make index-rag           # indexa filings no ChromaDB
make serve               # sobe FastAPI em http://localhost:8000
```

### Stack completa via Docker (API + Prometheus + Grafana + MLflow)

```bash
docker-compose up --build -d
# Swagger:    http://localhost:8000/docs
# Grafana:    http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# MLflow UI:  http://localhost:5000
```

### Hit no agente

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Considerando o ultimo 10-K da Apple, o preco atual e a projecao do LSTM, qual seu sumario sobre comprar AAPL hoje?"}'
```

Resposta tipica (resumida):

```json
{
  "resposta": "Apple esta cotada a USD 230 (yfinance). LSTM projeta USD 233 em 1 dia. O 10-K aponta crescimento de servicos como driver principal. Sentimento: neutro-positivo. Sumario: tendencia altista de curtissimo prazo, mas volatilidade alta. Esta resposta e informativa e nao constitui recomendacao de investimento.",
  "tools_chamadas": ["consultar_preco", "prever_preco_lstm", "buscar_em_filings", "analisar_sentimento"],
  "trace_id": "lf-..."
}
```

---

## Arquitetura

```
                         USUARIO (analista)
                         "Devo comprar AAPL?"
                                  |
                                  v
                  +-------------------------------+
                  |   API FastAPI                 |
                  |   POST /chat (principal)      |
                  |   POST /predict (legado F4)   |
                  |   GET /health, /metrics       |
                  +---------------+---------------+
                                  |
        +-------------------------+-------------------------+
        v                         v                         v
  Input Guardrail           Agente ReAct            Output Guardrail
  (regex inj +              (Gemini 2.0 Flash)      (Presidio PII)
   max length)              max_iter=10
                                  |
        +-------------+-----------+-----------+-------------+
        v             v           v           v             v
   consultar      prever        analisar   buscar_em      (futuro:
   _preco         _preco_lstm   _sentimento _filings      mais tools)
   (yfinance)     (PyTorch)     (sklearn)  (ChromaDB
                                            + RAG)

   +-----------------------------------------------------------+
   |  OBSERVABILIDADE (em paralelo)                            |
   |  MLflow (modelos) | Langfuse (traces LLM)                 |
   |  Prometheus + Grafana (metricas tecnicas)                 |
   |  Evidently (drift report offline)                         |
   +-----------------------------------------------------------+
```

Diagrama detalhado: ver `docs/SYSTEM_CARD.md`.

---

## Cobertura dos requisitos do Datathon

| Etapa | Entrega                                                                                           | Status |
|-------|---------------------------------------------------------------------------------------------------|--------|
| 1. Dados + Baseline                | LSTM PyTorch + Sentimento sklearn + MLflow tracking + schemas pandera     | OK     |
| 2. LLM + Agente + RAG              | ReAct Gemini + 4 tools + ChromaDB com filings reais + 3 configs benchmark | OK     |
| 3. Avaliacao + Observabilidade     | RAGAS 4 metricas + LLM-judge 3 criterios + Langfuse + Grafana             | OK     |
| 4. Seguranca + Governanca          | Guardrails I/O + OWASP Top10 LLM (5) + Red Team (5) + Cards + LGPD        | OK     |

### Cobertura dos 9 GAPs do Datathon

| #  | GAP                                  | Status               |
|----|--------------------------------------|----------------------|
| 01 | Ausencia de monitoramento            | Total                |
| 02 | Notebook como SPOF                   | Total                |
| 03 | Feature store destrutivo             | Parcial / por design |
| 04 | Cobertura de testes ~0               | Total (72 testes)    |
| 05 | Sem governanca de versionamento      | Total                |
| 06 | Sem deteccao de drift                | Minimo               |
| 07 | Retraining manual                    | Por design           |
| 08 | Dev sem dados                        | Total                |
| 09 | Skills gap eng. software             | Total                |

Detalhes e justificativas dos parciais: `docs/SYSTEM_CARD.md`.

---

## Documentos

| Documento                                                  | Conteudo                                            |
|------------------------------------------------------------|-----------------------------------------------------|
| [Model Card](docs/MODEL_CARD.md)                           | LSTM e sentimento — dados, metricas, limitacoes     |
| [System Card](docs/SYSTEM_CARD.md)                         | Arquitetura, GAPs, trade-offs, riscos residuais     |
| [LGPD Plan](docs/LGPD_PLAN.md)                             | Bases legais, direitos do titular, DPO, retencao    |
| [OWASP Mapping](docs/OWASP_MAPPING.md)                     | OWASP Top 10 para LLM Apps — mitigacoes implementadas |
| [Red Team Report](docs/RED_TEAM_REPORT.md)                 | 5 cenarios adversariais executados                   |
| [Pitch slides](docs/PITCH.md)                              | Slides Marp pra apresentacao                         |
| [Roteiro do video](docs/INSTRUCOES_VIDEO.md)               | Como gravar a demo de 8-10 min                       |
| [Design spec](docs/superpowers/specs/2026-04-26-datathon-fase05-design.md) | Spec completo (~1500 linhas)         |
| [Plano de implementacao](docs/superpowers/plans/2026-04-26-datathon-fase05-implementacao.md) | Plano dia-a-dia |

---

## Comandos do Makefile

| Comando                  | Descricao                                                          |
|--------------------------|--------------------------------------------------------------------|
| `make install`           | Instala deps (`pip install -e ".[dev]"`)                           |
| `make test`              | pytest com cobertura `>=60%`                                       |
| `make lint`              | ruff check em `src/`, `tests/`, `evaluation/`                      |
| `make typecheck`         | mypy com `--ignore-missing-imports`                                |
| `make security`          | bandit (severidade media+)                                         |
| `make train-lstm`        | Treina LSTM AAPL e loga no MLflow                                  |
| `make train-sentiment`   | Treina classificador de sentimento                                 |
| `make download-filings`  | Baixa 10-K/10-Q da SEC EDGAR                                       |
| `make index-rag`         | Indexa filings no ChromaDB                                         |
| `make serve`             | Sobe FastAPI (`uvicorn src.serving.app:app --reload --port 8000`) |
| `make mlflow-ui`         | Sobe MLflow UI em `:5000`                                          |
| `make eval`              | Roda RAGAS (4 metricas)                                            |
| `make benchmark`         | Roda 3 configs do agente em paralelo                               |
| `make drift`             | Gera relatorio Evidently em `evaluation/results/drift/`            |
| `make smoke`             | Smoke test manual (sobe stack + curl `/health`)                    |
| `make clean`             | Remove caches (`mlruns`, `chroma_db`, `.pytest_cache`, etc)        |

---

## Endpoints da API

| Metodo | Path        | Descricao                                                   |
|--------|-------------|-------------------------------------------------------------|
| GET    | `/health`   | Health check (status da API e dos modelos carregados)       |
| POST   | `/chat`     | Agente ReAct (endpoint principal do Datathon Fase 5)        |
| POST   | `/predict`  | LSTM direto (legado da Fase 4, mantido por compatibilidade) |
| GET    | `/metrics`  | Metricas no formato Prometheus                              |
| GET    | `/docs`     | Swagger UI (gerado automaticamente)                         |

Schemas Pydantic em `src/serving/app.py`.

---

## Estrutura do projeto (resumida)

```
phase-5/
├── README.md                    (este arquivo)
├── Makefile                     (atalhos de desenvolvimento)
├── Dockerfile                   (multi-stage — build + runtime)
├── docker-compose.yml           (api + prometheus + grafana + mlflow)
├── pyproject.toml
├── requirements.txt
├── .env.example                 (template; .env nao versionado)
│
├── src/
│   ├── data/                    (collectors yfinance + SEC EDGAR)
│   ├── features/                (schemas pandera + preprocessing)
│   ├── models/                  (LSTM PyTorch + train.py com MLflow)
│   ├── agent/                   (4 tools + RAG pipeline + ReAct agent)
│   ├── serving/                 (FastAPI app)
│   ├── security/                (input/output guardrails)
│   └── monitoring/              (Prometheus + Langfuse + Evidently)
│
├── evaluation/
│   ├── ragas_eval.py            (4 metricas RAGAS)
│   ├── llm_judge.py             (3 criterios LLM-as-judge)
│   ├── benchmark_configs.py     (3 configs em paralelo)
│   └── results/                 (JSON e HTML dos relatorios)
│
├── configs/                     (model_config.yaml + prompts.yaml)
├── monitoring/                  (prometheus.yml + grafana provisioning)
├── notebooks/                   (EDA — SEM logica de producao)
├── tests/                       (72 testes — pytest com cobertura ≥60%)
├── docs/                        (Model Card, System Card, LGPD, OWASP, etc)
└── models/                      (artefatos: lstm_torch.pt, sentiment.joblib)
```

---

## Video de demonstracao

**Link:** `(preencher apos gravacao)`

Roteiro e instrucoes de gravacao em [`docs/INSTRUCOES_VIDEO.md`](docs/INSTRUCOES_VIDEO.md).

---

## Historico — Fase 4

Este projeto reusa ~30% do codigo da Fase 4 (LSTM AAPL com TensorFlow/Keras).
A Fase 4 entregou um modelo standalone com API `/predict`. Aqui esse modelo
virou uma **tool** (`prever_preco_lstm`) chamada pelo agente. O endpoint
`/predict` segue disponivel pra compatibilidade. A versao em PyTorch passou a
ser o caminho oficial; a versao Keras ainda esta em `models/lstm_model.keras`
como referencia historica.

---

## Convencoes de Codigo

- Codigo, comentarios e docstrings em **portugues** (alinhado com a apresentacao)
- Type hints em todas as funcoes publicas
- Logging estruturado (sem `print`)
- Testes unitarios com pytest (`tests/test_*.py`)
- Lint: ruff | Type: mypy | Security: bandit
- Sem secrets versionados (vide `.env.example`)
