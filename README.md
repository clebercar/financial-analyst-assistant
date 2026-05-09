# Financial Analyst Assistant

[![tests](https://img.shields.io/badge/tests-72%20passing-brightgreen)]()
[![coverage](https://img.shields.io/badge/coverage-%E2%89%A560%25-blue)]()
[![python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

Assistente conversacional para analistas de buy-side decidirem compra/venda de
acoes. Combina **RAG sobre filings 10-K/10-Q da SEC**, **previsao com LSTM**,
**classificacao de sentimento** e **consulta de precos em tempo real** via um
agente ReAct (Gemini 2.5 Flash) — com observabilidade, guardrails de seguranca
e governanca documentada.

> **TL;DR:** o usuario pergunta `"Devo comprar AAPL hoje?"` no `POST /chat`. O
> agente decide quais ferramentas chamar (preco, projecao, filing, sentimento),
> compoe um sumario citavel e aplica guardrails de PII na saida. Tudo
> instrumentado em Langfuse, Prometheus e Grafana.

---

## Stack

| Camada           | Tecnologia                                                        |
|------------------|-------------------------------------------------------------------|
| Deep Learning    | PyTorch 2.x (LSTM)                                                |
| ML classico      | scikit-learn (TF-IDF + Logistic Regression)                       |
| Agente / LLM     | LangChain + LangGraph ReAct + Gemini 2.5 Flash                    |
| RAG              | ChromaDB + Gemini embeddings + filings da SEC EDGAR               |
| API              | FastAPI + Uvicorn                                                 |
| Tracking         | MLflow (experimentos + model registry)                            |
| Tracing LLM      | Langfuse (custos, latencia por step, faithfulness)                |
| Metricas         | Prometheus + Grafana                                              |
| Drift            | Evidently (PSI offline)                                           |
| Seguranca        | Regex anti-injection + Microsoft Presidio (PII redaction)         |
| Container        | Docker + docker-compose                                           |
| Qualidade        | pytest, ruff, mypy, bandit, pandera                               |

---

## Quick start

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
# editar .env e preencher GEMINI_API_KEY (e Langfuse keys, se quiser tracing)

make install
```

### Pipeline completo (treino -> indexacao -> serve)

```bash
make train-lstm          # treina LSTM e loga no MLflow
make train-sentiment     # treina classificador de sentimento
make download-filings    # baixa 10-K/10-Q da SEC EDGAR
make index-rag           # indexa filings no ChromaDB (~20 min: rate limit free tier)
make serve               # sobe FastAPI em http://localhost:8000
```

### Retreino via CI/CD (trigger manual)

Workflow `.github/workflows/retrain.yml` retreina os modelos sob demanda
direto pelo GitHub Actions, com gate de qualidade automatico:

1. Aba **Actions** -> workflow **Retrain** -> botao **Run workflow**
2. Inputs:
   - `model`: `lstm`, `sentiment` ou `all` (matrix paralela)
   - `ticker`: ticker do LSTM (default `AAPL`)
   - `start_date` / `end_date`: janela historica (vazio = usa `configs/model_config.yaml`)
3. O job treina, compara metricas com os `thresholds` do config
   (`mae_max`, `rmse_max`, `mape_max` para LSTM; `accuracy_min`, `f1_macro_min`
   para sentiment) via `src/models/evaluate_gate.py` e falha se violar.
4. Artefatos (`*.pt`, `*.joblib`, `metrics_<model>.json`, `mlruns/`) ficam
   disponiveis pra download por 30 dias na aba do run, mesmo se o gate falhar.

Para retreinar localmente com override de ticker:

```bash
python -m src.models.train --model lstm --ticker GOOGL
python -m src.models.evaluate_gate --model lstm
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
  "iteracoes": 4
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
                  |   POST /chat                  |
                  |   GET /health, /metrics       |
                  +---------------+---------------+
                                  |
        +-------------------------+-------------------------+
        v                         v                         v
  Input Guardrail           Agente ReAct            Output Guardrail
  (regex anti-injection +   (Gemini 2.5 Flash)      (Presidio PII)
   max length 4096)         max_iter=10
                                  |
        +-------------+-----------+-----------+-------------+
        v             v           v           v
   consultar      prever        analisar   buscar_em
   _preco         _preco_lstm   _sentimento _filings
   (yfinance)     (PyTorch)     (sklearn)   (ChromaDB
                                             + RAG)

   +-----------------------------------------------------------+
   |  OBSERVABILIDADE (em paralelo)                            |
   |  MLflow (modelos) | Langfuse (traces LLM)                 |
   |  Prometheus + Grafana (metricas tecnicas)                 |
   |  Evidently (drift report offline)                         |
   +-----------------------------------------------------------+
```

Diagrama detalhado em `docs/SYSTEM_CARD.md`.

---

## Resultados de avaliacao

Numeros gerados por `make smoke`, `make eval`, `make benchmark` e
`python -m evaluation.llm_judge`. Persistidos em `evaluation/results/`.

### Smoke test E2E (agente real, 7 perguntas)

7 / 7 sucesso. Pergunta multi-hop usou 4 tools em sequencia
(`buscar_em_filings -> consultar_preco -> prever_preco_lstm -> buscar_em_filings`)
em 14.6 s.

### Benchmark de 3 configuracoes

| Config             | Modelo                  | top_k | Latencia media |
|--------------------|-------------------------|-------|----------------|
| A — baseline       | `gemini-2.5-flash`      | 3     | 7.2 s          |
| B — mais contexto  | `gemini-2.5-flash`      | 5     | 21.4 s         |
| C — modelo menor   | `gemini-2.5-flash-lite` | 3     | 2.4 s          |

### RAGAS (golden set 20 itens, 4 metricas)

| Metrica            | Score |
|--------------------|-------|
| answer_relevancy   | 0.715 |
| faithfulness       | 0.254 |
| context_precision  | 0.308 |
| context_recall     | 0.146 |

Discussao do `faithfulness` aplicado a agentes multi-tool em
`docs/SYSTEM_CARD.md`.

### LLM-as-judge (3 criterios, escala 0-5, n=20)

| Criterio              | Score | Tipo    |
|-----------------------|-------|---------|
| coerencia_tecnica     | 4.55  | tecnico |
| completude            | 3.88  | tecnico |
| citacao_fontes (KPI)  | 3.25  | negocio |

---

## Documentos

| Documento                                  | Conteudo                                              |
|--------------------------------------------|-------------------------------------------------------|
| [Model Card](docs/MODEL_CARD.md)           | LSTM e sentimento — dados, metricas, limitacoes       |
| [System Card](docs/SYSTEM_CARD.md)         | Arquitetura, decisoes, trade-offs, riscos residuais   |
| [LGPD Plan](docs/LGPD_PLAN.md)             | Bases legais, direitos do titular, retencao           |
| [OWASP Mapping](docs/OWASP_MAPPING.md)     | OWASP Top 10 para LLM Apps — mitigacoes implementadas |
| [Red Team Report](docs/RED_TEAM_REPORT.md) | Cenarios adversariais executados                      |

---

## Comandos do Makefile

| Comando                  | Descricao                                                          |
|--------------------------|--------------------------------------------------------------------|
| `make install`           | Instala deps (`pip install -e ".[dev]"`)                           |
| `make test`              | pytest com cobertura `>=60%`                                       |
| `make lint`              | ruff check em `src/`, `tests/`, `evaluation/`                      |
| `make typecheck`         | mypy com `--ignore-missing-imports`                                |
| `make security`          | bandit (severidade media+)                                         |
| `make train-lstm`        | Treina LSTM e loga no MLflow                                       |
| `make train-sentiment`   | Treina classificador de sentimento                                 |
| `make download-filings`  | Baixa 10-K/10-Q da SEC EDGAR                                       |
| `make index-rag`         | Indexa filings no ChromaDB                                         |
| `make serve`             | Sobe FastAPI (`uvicorn src.serving.app:app --reload --port 8000`) |
| `make mlflow-ui`         | Sobe MLflow UI em `:5000`                                          |
| `make eval`              | Roda RAGAS (4 metricas)                                            |
| `make benchmark`         | Compara 3 configuracoes do agente                                  |
| `make drift`             | Gera relatorio Evidently em `evaluation/results/drift/`            |
| `make smoke`             | Smoke test E2E (5 queries + 2 cenarios red team contra agente real)|
| `make clean`             | Remove caches (`mlruns`, `chroma_db`, `.pytest_cache`, etc)        |

---

## Endpoints da API

| Metodo | Path        | Descricao                                                   |
|--------|-------------|-------------------------------------------------------------|
| GET    | `/health`   | Health check                                                |
| POST   | `/chat`     | Agente ReAct (endpoint principal)                           |
| GET    | `/metrics`  | Metricas no formato Prometheus                              |
| GET    | `/docs`     | Swagger UI (gerado automaticamente)                         |

Schemas Pydantic em `src/serving/schemas.py`.

---

## Estrutura do projeto

```
phase-5/
├── README.md                    (este arquivo)
├── Makefile                     (atalhos de desenvolvimento)
├── Dockerfile                   (multi-stage — build + runtime)
├── docker-compose.yml           (api + prometheus + grafana + mlflow)
├── pyproject.toml
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
│   ├── benchmark_configs.py     (3 configs comparadas)
│   └── results/                 (JSON e HTML dos relatorios)
│
├── configs/                     (model_config.yaml + prompts.yaml)
├── monitoring/                  (prometheus.yml + grafana provisioning)
├── notebooks/                   (EDA — sem logica de producao)
├── tests/                       (pytest unitario com cobertura ≥60%)
├── docs/                        (Model Card, System Card, LGPD, OWASP, etc)
└── models/                      (artefatos: lstm_torch.pt, sentiment.joblib)
```

---

## Convencoes de codigo

- Codigo, comentarios e docstrings em **portugues**
- Type hints em todas as funcoes publicas
- Logging estruturado (sem `print`)
- Testes unitarios com pytest (`tests/test_*.py`)
- Lint: ruff | Type: mypy | Security: bandit
- Sem secrets versionados (vide `.env.example`)
