# System Card - Assistente de Analista Financeiro

## 1. Visao Geral

Assistente conversacional que ajuda analistas de buy-side a avaliar acoes
combinando RAG sobre filings 10-K/10-Q da SEC, consulta de precos via
Yahoo Finance, previsao com modelo LSTM, e classificacao de sentimento via
TF-IDF + Logistic Regression. Orquestrado por agente ReAct com Gemini 2.0
Flash, exposto por API FastAPI com observabilidade ponta-a-ponta.

### Diagrama de arquitetura (texto)

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

## 2. Componentes

| Componente   | Tecnologia                          | Responsabilidade                  |
|--------------|-------------------------------------|-----------------------------------|
| API          | FastAPI                             | Servir endpoints HTTP             |
| Agente       | LangChain ReAct + Gemini 2.0 Flash  | Orquestrar tools                  |
| RAG          | ChromaDB + Gemini embeddings        | Retrieval de filings              |
| Tools (4)    | yfinance, PyTorch LSTM, sklearn, RAG| Capacidades especificas           |
| Tracking     | MLflow                              | Registro de experimentos          |
| Tracing LLM  | Langfuse                            | Observabilidade do agente         |
| Metricas     | Prometheus + Grafana                | Operacional                       |
| Drift        | Evidently                           | Estabilidade de features          |
| Guardrails   | regex + Presidio                    | Seguranca I/O                     |

## 3. Cobertura dos 9 GAPs do Datathon

| #  | GAP do guia                          | Cobertura            | Como demonstramos                                                                                  |
|----|--------------------------------------|----------------------|----------------------------------------------------------------------------------------------------|
| 01 | Ausencia de monitoramento            | Total                | Prometheus + Langfuse + Grafana                                                                    |
| 02 | Notebook como SPOF                   | Total                | Codigo modular em `src/`, notebooks so EDA                                                         |
| 03 | Feature store destrutivo             | Parcial / por design | Nao temos feature store (justificado nessa Secao 4.2 como consciente do anti-padrao)               |
| 04 | Cobertura de testes ~0               | Total                | pytest `--cov-fail-under=60`, schemas pandera                                                      |
| 05 | Sem governanca de versionamento      | Total                | MLflow com schema obrigatorio de tags + Model Card                                                 |
| 06 | Sem deteccao de drift                | Minimo               | Relatorio Evidently offline (nao retrigger automatico)                                             |
| 07 | Retraining manual                    | Por design           | Champion-challenger descrito nessa Secao 4.3, nao implementado                                     |
| 08 | Dev sem dados                        | Total                | Fixtures sinteticos + dados publicos                                                               |
| 09 | Skills gap eng. software             | Total                | Type hints, docstrings, logging estruturado, pyproject.toml, ruff/mypy/bandit                      |

**Estrategia para os 3 parciais (03, 06, 07):** honestidade tecnica neste
System Card. Em vez de implementacao superficial, descrevemos o problema, a
abordagem proposta e por que ficou fora do MVP de 9 dias.

## 4. Trade-offs e Decisoes

### 4.1 LLM hosted (Gemini API) vs self-hosted quantizado

**Decisao:** hosted via API.

**Por que:** self-hosting + quantizacao (vLLM/BentoML) tomaria 1+ semana
sozinho. MVP precisa caber em 9 dias.

**Roadmap pos-MVP:** servir Llama-3.1-8B quantizado 4-bit via vLLM, comparar
faithfulness e custo com Gemini hosted. Configuracao inicial:

```yaml
vllm:
  model: meta-llama/Meta-Llama-3.1-8B-Instruct
  quantization: awq
  gpu_memory_utilization: 0.9
```

### 4.2 Sem feature store (cobre GAP 03)

**Decisao:** nao implementar.

**Por que:** projeto tem 2 modelos com features simples (precos
normalizados, TF-IDF). Feature store agregaria overhead sem ROI no MVP.

**Estamos cientes do anti-padrao:** se escalassemos para >10 modelos,
implementariamos com upsert incremental (nunca FLUSHALL + bulk load - que
e o anti-padrao classico que o Datathon menciona).

### 4.3 Drift detection offline (cobre GAP 06 e 07)

**Decisao:** apenas relatorio Evidently sob demanda (`make drift`).

**Por que:** retrigger automatico exige champion-challenger pipeline
funcional - nao cabe no MVP.

**Roadmap (champion-challenger):**
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

### 4.4 Apenas testes unitarios

**Decisao consciente do autor:** sem integration tests automatizados.

**Trade-off:** maior risco de regressao silenciosa em integracoes externas
(Gemini API, yfinance, ChromaDB).

**Mitigacao:** `make smoke` rodado manualmente antes de cada release; CI
nao executa rede externa.

## 5. Roadmap Pos-MVP

1. **Champion-challenger retraining** com aprovacao humana antes de promover.
2. **Drift retrigger automatico** (PSI > 0.2 dispara retraining job).
3. **Quantizacao self-hosted** com vLLM (Llama-3.1-8B-Instruct AWQ).
4. **Rate limit por IP** com `slowapi` no `/chat` (cobre LLM10 totalmente).
5. **Decodificacao base64 antes do guardrail** (cobre Cenario 3 do red team).
6. **LSTM multi-ticker** treinado com transferencia (5+ empresas).
7. **Integration tests** end-to-end com VCR cassettes para mocks de rede.
8. **Llama Guard local** como segundo filtro de input.

## 6. Riscos Residuais

| Risco                                           | Probabilidade | Impacto | Mitigacao atual                              |
|-------------------------------------------------|---------------|---------|----------------------------------------------|
| Vazamento de PII em logs                        | Baixa         | Alto    | Output guardrail Presidio                    |
| Decisao automatizada por usuario inexperiente   | Media         | Alto    | Disclaimer explicito em todas as respostas   |
| Custo Gemini explode em producao                | Media         | Medio   | `max_tokens` + limite 4096 chars no input    |
| Filings desatualizados (nao re-indexados)       | Alta          | Baixo   | Re-indexacao manual via `make index-rag`     |
| Drift no LSTM (treino terminou em 2024-12-31)   | Alta          | Medio   | Relatorio Evidently sob demanda              |

## 7. Equipe e Responsabilidades

- **Owner:** Cleber Carvalho
- **DPO (LGPD):** Cleber Carvalho (interim)
- **On-call:** Cleber Carvalho

## 8. Conformidade

- LGPD: ver `docs/LGPD_PLAN.md`
- OWASP Top 10 LLM: ver `docs/OWASP_MAPPING.md`
- Red team: ver `docs/RED_TEAM_REPORT.md`
- Model Card: ver `docs/MODEL_CARD.md`
