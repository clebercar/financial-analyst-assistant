---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Datathon Fase 05 — MLET FIAP'
footer: 'Cleber Carvalho · 2026-05-05'
---

# Assistente de Analista Financeiro
### Datathon Fase 05 — MLET FIAP

**Cleber Carvalho** · 2026-05-05

`contatoclebercarvalho@gmail.com`

---

## Problema

- Analistas de buy-side gastam **horas** lendo um unico 10-K
- Decisoes precisam combinar: **documentos + precos + projecoes + sentimento**
- Bancas e fintechs precisam de assistentes **auditaveis** (LGPD, ANPD)
- Custo cognitivo alto, deadlines apertados

> **Pergunta:** da pra reduzir o tempo de analise sem perder rigor?

---

## Abordagem

```
USUARIO -> /chat -> [Input Guardrail] -> Agente ReAct (Gemini 2.5 Flash)
                                              |
              +-------------------+-----------+-----------+--------------------+
              v                   v                       v                    v
       consultar_preco     prever_preco_lstm     analisar_sentimento     buscar_em_filings
        (yfinance)          (PyTorch LSTM)         (sklearn TF-IDF)       (ChromaDB + RAG)
                                              |
                                              v
                                    [Output Guardrail (Presidio PII)]
                                              |
                                              v
                                          USUARIO
```

- **Agente ReAct** decide quais ferramentas chamar a cada turno
- **4 tools** especificas do dominio financeiro
- **RAG** sobre 10-K/10-Q da SEC indexados no ChromaDB
- **MLOps Level 2:** MLflow + Langfuse + Prometheus + Evidently + Presidio
- **~30% reuso da Fase 4** (LSTM AAPL vira tool plug-and-play)

---

## Stack

| Camada           | Tecnologia                                              |
|------------------|---------------------------------------------------------|
| Deep Learning    | PyTorch 2.x (LSTM)                                      |
| ML classico      | scikit-learn (sentimento)                               |
| Agente / LLM     | LangChain ReAct + Gemini 2.5 Flash                      |
| RAG              | ChromaDB + Gemini embeddings + SEC EDGAR                |
| API              | FastAPI + Uvicorn                                       |
| Tracking         | MLflow (modelos)                                        |
| Tracing          | Langfuse (LLM)                                          |
| Metricas         | Prometheus + Grafana                                    |
| Drift            | Evidently (PSI offline)                                 |
| Seguranca        | Regex anti-injection + Microsoft Presidio (PII)         |

---

## Demo (4 min)

1. **`/chat`** — pergunta multi-hop ("Devo comprar AAPL hoje?") -> agente chama 4 tools
2. **Langfuse trace** — ver cada step do ReAct, latencia e custo por tool
3. **Grafana dashboard** — metricas tecnicas (RPS, latencia P95, error rate)
4. **RAGAS scores** — `evaluation/results/ragas_scores.json` com 4 metricas
5. **Red team** — `curl` com prompt injection -> bloqueado com HTTP 400

---

## Resultados — Smoke test E2E (real)

`make smoke` rodou 5 perguntas + 2 cenarios red team contra agente real.
**7/7 sucesso.** Resultados em `evaluation/results/smoke_test.json`.

| Categoria          | Tools usadas (em ordem)                              | Iter | Tempo (s) |
|--------------------|------------------------------------------------------|------|-----------|
| RAG puro (10-K AAPL)| `buscar_em_filings`                                 | 1    | 7.1       |
| Tool simples (NVDA) | `consultar_preco`                                   | 1    | 2.8       |
| LSTM (AAPL 5d)      | `prever_preco_lstm`                                 | 1    | 3.6       |
| Sentimento          | `analisar_sentimento`                               | 1    | 3.0       |
| Multi-hop AAPL      | `buscar_em_filings -> consultar_preco -> LSTM -> ...`| 4    | 14.6      |
| Red team 4 (PII)    | recusa direta, sem tool                             | 0    | 1.1       |
| Red team 5 (loop)   | recusa direta, sem tool                             | 0    | 1.4       |

**RAG:** 300 chunks indexados (10 filings da SEC × 30 chunks/cada),
embeddings `gemini-embedding-001` (3072 dims), ChromaDB persistido.

**Testes automatizados:** 72 testes unitarios passando.

---

## Cobertura dos 9 GAPs do Datathon

| #  | GAP                                  | Status               |
|----|--------------------------------------|----------------------|
| 01 | Ausencia de monitoramento            | Total                |
| 02 | Notebook como SPOF                   | Total                |
| 03 | Feature store destrutivo             | Parcial / por design |
| 04 | Cobertura de testes ~0               | Total                |
| 05 | Sem governanca de versionamento      | Total                |
| 06 | Sem deteccao de drift                | Minimo               |
| 07 | Retraining manual                    | Por design           |
| 08 | Dev sem dados                        | Total                |
| 09 | Skills gap eng. software             | Total                |

**6 cobertos** / **3 parciais com justificativa** documentada no System Card.

---

## Seguranca + Governanca

- **Input guardrail:** regex anti-prompt-injection (7 padroes) + max length
- **Output guardrail:** Microsoft Presidio (PII redaction — PERSON, EMAIL, CPF, etc)
- **OWASP Top 10 LLM:** 5 ameacas mapeadas e mitigadas (LLM01, 02, 06, 07, 10)
- **Red team:** 5 cenarios executados — 4/5 bloqueados, 1 entra no roadmap
- **Model Card + System Card** preenchidos com metricas reais e GAPs
- **LGPD:** plano com bases legais, retencao, direitos do titular, DPO

---

## Roadmap pos-MVP

1. **Quantizacao self-hosted** com vLLM + Llama-3.1-8B AWQ (custo)
2. **Champion-challenger** automatico com aprovacao humana (governanca)
3. **Drift retrigger** automatico (PSI > 0.2 dispara retraining)
4. **Rate limit** por IP no `/chat` com `slowapi` (cobre LLM10 totalmente)
5. **Decodificacao base64** antes do guardrail (cobre Cenario 3 do red team)
6. **Integration tests** com VCR cassettes pra mocks de rede
7. **LSTM multi-ticker** com transferencia (5+ empresas)

---

## Obrigado

**Repo:** `github.com/cleber/postgraduate-machine-learning`

**Video:** `(preencher apos gravacao)`

**Docs:** `docs/MODEL_CARD.md` · `docs/SYSTEM_CARD.md` · `docs/OWASP_MAPPING.md`

**Contato:** `contatoclebercarvalho@gmail.com`

---

# Apendice — Como rodar

```bash
cp .env.example .env             # editar GEMINI_API_KEY
make install
make train-lstm
make train-sentiment
make download-filings
make index-rag
make serve

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Devo comprar AAPL hoje?"}'
```

Stack completa via Docker: `docker-compose up --build -d`.
