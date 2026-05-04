# System Card

## 1. Visao Geral

Assistente conversacional que ajuda analistas de buy-side a avaliar acoes
combinando RAG sobre filings 10-K/10-Q da SEC, consulta de precos via Yahoo
Finance, previsao com modelo LSTM e classificacao de sentimento via TF-IDF +
Logistic Regression. Orquestrado por agente ReAct com Gemini 2.5 Flash,
exposto por API FastAPI com observabilidade ponta-a-ponta.

### Diagrama de arquitetura

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

## 2. Componentes

| Componente   | Tecnologia                          | Responsabilidade                  |
|--------------|-------------------------------------|-----------------------------------|
| API          | FastAPI                             | Servir endpoints HTTP             |
| Agente       | LangGraph ReAct + Gemini 2.5 Flash  | Orquestrar tools                  |
| RAG          | ChromaDB + Gemini embeddings        | Retrieval de filings              |
| Tools (4)    | yfinance, PyTorch LSTM, sklearn, RAG| Capacidades especificas           |
| Tracking     | MLflow                              | Registro de experimentos          |
| Tracing LLM  | Langfuse                            | Observabilidade do agente         |
| Metricas     | Prometheus + Grafana                | Operacional                       |
| Drift        | Evidently                           | Estabilidade de features          |
| Guardrails   | regex + Presidio                    | Seguranca I/O                     |

## 3. Decisoes de Arquitetura

### 3.1 LLM hosted (Gemini API) vs self-hosted quantizado

**Decisao:** hosted via API.

**Motivacao:** complexidade operacional de manter LLM self-hosted (vLLM/BentoML
+ quantizacao + monitoramento de GPU) nao se justifica para o escopo do
sistema. Provedores hosted oferecem alta disponibilidade, atualizacoes
automaticas e custo previsivel para cargas baixas/medias.

**Quando reconsiderar:** se o volume de requests crescer ao ponto de o custo
mensal hosted ultrapassar o TCO de uma instancia GPU dedicada, ou se houver
requisito regulatorio de "dados nao saem do perimetro".

**Setup futuro proposto:**
```yaml
vllm:
  model: meta-llama/Meta-Llama-3.1-8B-Instruct
  quantization: awq
  gpu_memory_utilization: 0.9
```

### 3.2 Sem feature store

**Decisao:** persistencia direta via `joblib` para o classificador de
sentimento; sem store intermediaria de features.

**Motivacao:** com 2 modelos e features simples (precos normalizados,
TF-IDF), uma feature store agregaria latencia, custo de operacao e
overhead de schema sem reuso real.

**Anti-padrao a evitar quando escalar:** atualizacao destrutiva (FLUSHALL +
bulk load) — janela de store vazio causa decisoes erradas em producao. Se
esse projeto adotar feature store no futuro, deve usar upsert incremental
com TTL.

### 3.3 Drift detection offline

**Decisao:** relatorio Evidently sob demanda (`make drift`); sem retrigger
automatico.

**Motivacao:** retrigger automatico exige champion-challenger pipeline
funcional, que requer ambiente de producao com trafego real. Em ambiente de
desenvolvimento, o relatorio offline atende para auditoria periodica e
investigacao de incidentes.

**Pipeline futuro proposto:**
```
[drift detector] --PSI > 0.2--> [retrain job] --> [shadow inference]
                                                       |
                                  +--------------------+
                                  v
                       [comparacao com champion atual]
                                  |
                       [aprovacao humana via PR]
                                  |
                                  v
                       [promote challenger -> champion]
```

### 3.4 Apenas testes unitarios

**Decisao:** suite de testes unitarios com dependencias externas mockadas;
sem integration tests automatizados em CI.

**Motivacao:** os testes unitarios validam contratos e logica isoladamente,
rodam em segundos e nao consomem rate limit da Gemini API. Integration
tests reais exigiriam credenciais e injetariam custo/instabilidade no CI.

**Mitigacao:** o script `make smoke` faz validacao end-to-end manual contra
o agente real (5 queries representativas + 2 cenarios de red team) e
persiste o resultado em `evaluation/results/smoke_test.json`. Recomenda-se
rodar antes de cada release.

### 3.5 RAGAS aplicada a agente multi-tool

**Observacao empirica** (golden set 20 itens):
- `answer_relevancy`: 0.715
- `faithfulness`: 0.254
- `context_precision`: 0.308
- `context_recall`: 0.146

**Diagnostico:** RAGAS faithfulness assume que `answer` deve ser apoiado
pelos `contexts` recuperados via RAG. Mas o agente usa **4 tools** (preco
yfinance, LSTM, sentimento, RAG). Quando responde `"preco da NVDA e $198"`,
RAGAS marca como nao-suportado porque so ve os chunks RAG no contexto.

**Conclusao:** trata-se de artefato metodologico (avaliar agente
multi-tool com metrica desenhada para RAG puro). LLM-as-judge confirma
qualidade das respostas (coerencia tecnica 4.55/5, completude 3.88/5).

**Roadmap RAGAS:**
1. Customizar `_build_rag_rows` para incluir todas as observacoes de tools
   como contexts (nao apenas chunks RAG).
2. Indexar mais chunks por filing (50-100, em vez dos 30 atuais) ou usar
   parsing semantico que prioriza Item 1A (Risk Factors) e Item 7 (MD&A).

## 4. Roadmap

1. Champion-challenger retraining com aprovacao humana antes de promover.
2. Drift retrigger automatico (PSI > 0.2 dispara retraining job).
3. Quantizacao self-hosted com vLLM (Llama-3.1-8B-Instruct AWQ).
4. Rate limit por IP com `slowapi` no `/chat` (cobre LLM10 totalmente).
5. Decodificacao base64 antes do guardrail (cobre Cenario 3 do Red Team).
6. LSTM multi-ticker treinado com transferencia (5+ empresas).
7. Integration tests end-to-end com VCR cassettes.
8. Llama Guard local como segundo filtro de input.

## 5. Riscos Residuais

| Risco                                         | Probabilidade | Impacto | Mitigacao atual                              |
|-----------------------------------------------|---------------|---------|----------------------------------------------|
| Vazamento de PII em logs                      | Baixa         | Alto    | Output guardrail Presidio                    |
| Decisao automatizada por usuario inexperiente | Media         | Alto    | Disclaimer explicito em todas as respostas   |
| Custo Gemini explode em producao              | Media         | Medio   | `max_tokens` + limite 4096 chars no input    |
| Filings desatualizados (nao re-indexados)     | Alta          | Baixo   | Re-indexacao manual via `make index-rag`     |
| Drift no LSTM (treino terminou em 2024-12-31) | Alta          | Medio   | Relatorio Evidently sob demanda              |

## 6. Equipe e Responsabilidades

- **Owner:** ml-team
- **DPO (LGPD):** ml-team (interim)
- **On-call:** ml-team

## 7. Conformidade

- LGPD: ver `docs/LGPD_PLAN.md`
- OWASP Top 10 LLM: ver `docs/OWASP_MAPPING.md`
- Red team: ver `docs/RED_TEAM_REPORT.md`
- Model Card: ver `docs/MODEL_CARD.md`
