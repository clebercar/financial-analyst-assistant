# Design — Datathon Fase 05: Assistente de Analista Financeiro

**Data:** 2026-04-26
**Autor:** Cleber (com apoio do Claude)
**Prazo:** 2026-05-05
**Status:** Aprovado pelo autor (brainstorming concluído)

---

## 1. Visão geral

### 1.1 Pitch
Assistente conversacional que ajuda analistas de buy-side a decidir compra/venda
de ações combinando: RAG sobre relatórios oficiais (10-K e 10-Q da SEC),
consulta de preços ao vivo (yfinance), previsão de preços via LSTM (reuso da
Fase 4) e classificação de sentimento de earnings (modelo baseline sklearn).

### 1.2 Caso de negócio (KPI alvo)
- **Problema:** analistas gastam horas lendo um único relatório anual (10-K).
- **Métrica de negócio:** redução do tempo médio de análise por empresa.
- **KPI técnico ligado:** *citation rate* — fração de respostas que cita a fonte
  (filing + data dos preços). Critério de confiabilidade pra uso em produção
  regulada.

### 1.3 Público
- **Persona:** analista buy-side iniciante a pleno
- **Não é pra:** decisão automatizada de compra/venda (fica explícito em todas
  as respostas)

---

## 2. Contexto e restrições

### 2.1 Restrições do projeto
| Restrição | Implicação |
|-----------|-----------|
| Solo (1 pessoa) | Escopo de equipe de 4-5 reduzido a MVP |
| 9 dias úteis (até 2026-05-05) | Cortes agressivos vs especificação original |
| Conhecimento técnico limitado do autor | Claude escreve quase 100% do código |
| Sem hardware GPU | Sem self-host de LLM quantizado |

### 2.2 Premissas
- Reuso da Fase 4 (LSTM + FastAPI + Prometheus + Docker) é **estratégico** pro
  pitch ("integro fases anteriores")
- Domínio mantido em finanças (alinha com foco do guia do Datathon)
- Gemini API (free tier) é viável pro volume do projeto

---

## 3. Cobertura dos 9 GAPs do Datathon

| # | GAP do guia | Cobertura | Como demonstramos |
|---|-------------|-----------|-------------------|
| 01 | Ausência de monitoramento | ✅ Total | Prometheus + Langfuse + Grafana |
| 02 | Notebook como SPOF | ✅ Total | Código modular em `src/`, notebooks só EDA |
| 03 | Feature store destrutivo | ⚠️ Parcial / por design | Não temos feature store (justificado no System Card como consciente do anti-padrão) |
| 04 | Cobertura de testes ~0 | ✅ Total | pytest `--cov-fail-under=60`, schemas pandera |
| 05 | Sem governança de versionamento | ✅ Total | MLflow com schema obrigatório de tags + Model Card |
| 06 | Sem detecção de drift | ⚠️ Mínimo | Relatório Evidently offline (não retrigger automático) |
| 07 | Retraining manual | ⚠️ Por design | Champion-challenger descrito no System Card, não implementado |
| 08 | Dev sem dados | ✅ Total | Fixtures sintéticos + dados públicos |
| 09 | Skills gap eng. software | ✅ Total | Type hints, docstrings, logging estruturado, pyproject.toml |

**Estratégia para os 3 parciais (03, 06, 07):** honestidade técnica no
System Card. Em vez de implementação superficial, descrevemos o problema, a
abordagem proposta e por que ficou fora do MVP de 9 dias.

---

## 4. Arquitetura

### 4.1 Diagrama de fluxo (texto)

```
                         USUÁRIO (analista)
                         "Devo comprar AAPL?"
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   API FastAPI                 │
                  │   POST /chat (principal)      │
                  │   POST /predict (legado F4)   │
                  │   GET /health, /metrics       │
                  └───────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  Input Guardrail           Agente ReAct            Output Guardrail
  (regex inj +              (Gemini 2.0 Flash)      (Presidio PII)
   max length)              max_iter=10
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┐
        ▼             ▼           ▼           ▼             ▼
   consultar      prever        analisar   buscar_em      (futuro:
   _preco         _preco_lstm   _sentimento _filings      mais tools)
   (yfinance)     (PyTorch)     (sklearn)  (ChromaDB
                                            + RAG)

   ┌─────────────────────────────────────────────────────────────┐
   │  OBSERVABILIDADE (em paralelo)                              │
   │  MLflow (modelos) | Langfuse (traces LLM)                   │
   │  Prometheus + Grafana (métricas técnicas)                   │
   │  Evidently (drift report offline)                           │
   └─────────────────────────────────────────────────────────────┘
```

### 4.2 Camadas e responsabilidades

| Camada | Responsabilidade | Tecnologia |
|--------|------------------|------------|
| Coleta de dados | Buscar dados públicos | yfinance, SEC EDGAR API, HuggingFace |
| ML clássico | Baseline preditivo (Etapa 1) | PyTorch (LSTM), scikit-learn (sentimento) |
| Tracking de experimentos | Registrar runs com metadata padronizada | MLflow local |
| LLM | Gerar respostas e raciocinar | Gemini 2.0 Flash (free tier API) |
| Embeddings | Vetorizar texto pro RAG | Gemini text-embedding-004 |
| Vector store | Indexar e buscar documentos | ChromaDB (persistido em disco) |
| Agente | Orquestrar tools com ReAct | LangChain `create_react_agent` |
| API | Servir endpoints HTTP | FastAPI (reuso Fase 4) |
| Tracing LLM | Capturar custo/latência/qualidade | Langfuse free tier |
| Métricas técnicas | Operacional | Prometheus + Grafana |
| Drift | Estabilidade de dados | Evidently (offline) |
| Segurança | Guardrails I/O | regex + Presidio |
| Container | Empacotamento | Docker + docker-compose |

---

## 5. Estrutura do repositório

Reuso da Fase 4 marcado com 🔄 (sem mudar), ♻️ (adaptar), 🆕 (novo).

```
phase-5/
├── .github/workflows/ci.yml                  🆕
├── data/
│   ├── raw/                                  🆕 (gitignore)
│   ├── processed/                            🆕
│   ├── filings/                              🆕 textos da SEC
│   └── golden_set/golden_set.json            🆕
├── src/
│   ├── data/
│   │   ├── collector.py                      🔄 yfinance
│   │   ├── sec_edgar.py                      🆕
│   │   └── financial_phrasebank.py           🆕
│   ├── features/feature_engineering.py       🆕
│   ├── models/
│   │   ├── lstm_torch.py                     ♻️ converter TF→PyTorch
│   │   ├── preprocessing.py                  🔄
│   │   ├── sentiment_classifier.py           🆕
│   │   └── train.py                          🆕 pipeline + MLflow
│   ├── agent/
│   │   ├── react_agent.py                    🆕
│   │   ├── tools.py                          🆕
│   │   └── rag_pipeline.py                   🆕
│   ├── serving/
│   │   ├── app.py                            ♻️ era src/api/main.py
│   │   ├── schemas.py                        🔄 + novos schemas
│   │   └── Dockerfile                        🔄
│   ├── monitoring/
│   │   ├── prometheus_metrics.py             ♻️ era src/api/monitoring.py
│   │   ├── langfuse_tracer.py                🆕
│   │   └── drift_report.py                   🆕
│   └── security/
│       ├── input_guardrail.py                🆕
│       └── output_guardrail.py               🆕
├── tests/
│   ├── conftest.py                           🆕 fixtures
│   ├── test_preprocessing.py                 🔄
│   ├── test_models.py                        🆕
│   ├── test_agent.py                         🆕 (LLM mockado)
│   ├── test_api.py                           ♻️ (deps mockadas)
│   ├── test_guardrails.py                    🆕
│   └── test_features.py                      🆕
├── evaluation/
│   ├── ragas_eval.py                         🆕
│   ├── llm_judge.py                          🆕
│   └── benchmark_configs.py                  🆕
├── docs/
│   ├── MODEL_CARD.md                         🆕
│   ├── SYSTEM_CARD.md                        🆕
│   ├── LGPD_PLAN.md                          🆕
│   ├── OWASP_MAPPING.md                      🆕
│   ├── RED_TEAM_REPORT.md                    🆕
│   └── superpowers/specs/                    🆕 (este arquivo)
├── notebooks/
│   ├── 01_eda_financial_phrasebank.ipynb     🆕
│   └── exploratory_analysis.ipynb            🔄 EDA AAPL Fase 4
├── configs/
│   ├── model_config.yaml                     🆕
│   └── prompts.yaml                          🆕 prompts versionados
├── models/
│   ├── lstm_torch.pt                         ♻️ migra .keras → .pt
│   ├── scaler.joblib                         🔄
│   └── sentiment_classifier.joblib           🆕
├── docker-compose.yml                        ♻️
├── pyproject.toml                            🆕 substitui requirements.txt
├── requirements.txt                          🔄 mantido pra compat
├── Makefile                                  🆕
├── .env.example                              🆕
├── .pre-commit-config.yaml                   🆕
├── CLAUDE.md                                 ♻️ atualiza pra Fase 5
└── README.md                                 ♻️ reescreve
```

**Reuso real da Fase 4:** ~30-40% do código. O essencial: a história do pitch
("LSTM da Fase 4 vira tool do agente") fica honesta.

---

## 6. Stack técnico (decisões)

| Aspecto | Escolha | Justificativa |
|---------|---------|---------------|
| Framework ML clássico | PyTorch (LSTM) + sklearn (sentimento) | Cumpre requisito explícito do guia (PyTorch + MLflow = 5%) |
| LLM | Gemini 2.0 Flash via API | Free tier generoso, autor escolheu |
| Embedding | Gemini text-embedding-004 | Stack consistente, sem hardware |
| Vector store | ChromaDB persistido | Mais simples possível pro escopo |
| Agente | LangChain `create_react_agent` | Padrão didático |
| API | FastAPI (Fase 4) | Sem reinventar |
| Container | Docker + docker-compose (Fase 4) | Sem reinventar |
| Tracing LLM | Langfuse free tier | Recomendado pelo guia |
| Métricas técnicas | Prometheus + Grafana | Reuso Fase 4 + dashboard |
| Drift | Evidently | Recomendado pelo guia |
| PII | Presidio | Recomendado pelo guia |
| Lint/Type | ruff + mypy + bandit | Padrão moderno |
| Testes | pytest (apenas unitários) | Decisão do autor |

---

## 7. Cronograma

Hoje: 2026-04-26 (domingo). Entrega: 2026-05-05.

| Dia | Data | Foco | Rota de fuga se atrasar |
|-----|------|------|-------------------------|
| 1 | 26/04 dom | Setup + esqueleto + MLflow local | (sem corte) |
| 2 | 27/04 seg | LSTM TF→PyTorch + MLflow tracking | Pular grid de hiperparâmetros |
| 3 | 28/04 ter | Sentimento sklearn + EDA + pandera | Pular EDA visual |
| 4 | 29/04 qua | RAG (SEC EDGAR + ChromaDB) + tools yfinance | Reduzir 10 → 3 filings |
| 5 | 30/04 qui | Agente ReAct + endpoint /chat + testes mock | Pular testes do agente |
| 6 | 01/05 sex 🇧🇷 | Golden set (20 pares) + RAGAS + LLM-as-judge | Reduzir 20 → 15 pares |
| 7 | 02/05 sáb | Langfuse + Grafana + Evidently | Pular Grafana |
| 8 | 03/05 dom | Guardrails + OWASP + red team + Cards + LGPD + CI | LGPD curto (1 página) |
| 9 | 04/05 seg | README + slides + script + vídeo + tag | Slides simples |
| DL | 05/05 ter | Buffer + entrega | — |

**Críticos (não atrasar):** Dia 2 (LSTM), Dias 4-5 (RAG/agente), Dia 9 (vídeo).

**Tarefas que requerem o autor (não delegáveis ao Claude):**
- Confirmar Gemini API key funcionando (Dia 1)
- Revisar e curar golden set (Dia 6) — ~1-2 horas
- Gravar o vídeo (Dia 9)
- Apresentação síncrona se exigida

---

## 8. Detalhes técnicos

### 8.1 Tools do agente (4 ferramentas)

```python
def consultar_preco(ticker: str) -> dict:
    """Preço atual + variação 30d.
    
    Returns:
        {'ticker': str, 'preco_atual': float, 'moeda': str,
         'variacao_30d_pct': float, 'volume_medio': float,
         'timestamp': str}
    """

def prever_preco_lstm(ticker: str, dias: int = 5) -> dict:
    """Projeção LSTM dos próximos N dias úteis.
    
    Returns:
        {'ticker': str, 'previsoes': list[dict],
         'metricas_modelo': {'mae': float, 'rmse': float, 'mape': float},
         'modelo_versao': str,
         'aviso': str}  # ex: "Modelo treinado apenas em AAPL"
    """

def analisar_sentimento(texto: str) -> dict:
    """Classifica sentimento de trecho de earnings/notícia.
    
    Returns:
        {'sentimento': 'positive' | 'neutral' | 'negative',
         'confianca': float}
    """

def buscar_em_filings(query: str,
                     ticker: str | None = None,
                     top_k: int = 3) -> dict:
    """RAG sobre 10-K e 10-Q indexados.
    
    Returns:
        {'chunks': list[{'ticker', 'tipo', 'ano', 'secao', 'trecho'}]}
    """
```

**Decisões importantes:**
- `prever_preco_lstm` retorna `aviso` quando ticker ≠ AAPL (LSTM treinado só em
  AAPL). Agente decide se usa.
- `buscar_em_filings` permite filtrar por ticker → reduz ruído no retrieval.
- Todas retornam `dict` (não string) → LangChain formata + fácil testar.

### 8.2 Configuração do RAG

**Documentos indexados:** 5 empresas × 2 docs = **10 filings**

| Empresa | 10-K (anual mais recente) | 10-Q (trimestre mais recente) |
|---------|---------------------------|-------------------------------|
| AAPL | ✅ | ✅ |
| MSFT | ✅ | ✅ |
| GOOGL | ✅ | ✅ |
| NVDA | ✅ | ✅ |
| META | ✅ | ✅ |

**Chunking:**
| Parâmetro | Valor |
|-----------|-------|
| Tamanho do chunk | 800 tokens |
| Overlap | 100 tokens |
| Embedding | Gemini text-embedding-004 (768 dims) |
| top_k default | 3 |
| Metadata | ticker, tipo, ano fiscal, seção |

### 8.3 Golden set (20 pares)

**Distribuição por categoria:**

| Categoria | Quantidade | Exemplo |
|-----------|------------|---------|
| RAG puro | 8 | "Quais os principais riscos da Apple no último 10-K?" |
| Tool simples | 4 | "Qual o preço atual da NVDA?" |
| Tool LSTM | 3 | "Qual sua previsão pra AAPL nos próximos 5 dias?" |
| Multi-hop | 5 | "Considerando 10-K, preço atual e LSTM, devo comprar AAPL?" |

**Estrutura de cada item:**

```json
{
  "id": "gs_001",
  "query": "string",
  "expected_answer": "string",
  "expected_contexts": ["chunk_id_1", "chunk_id_2"],
  "category": "rag_pure|tool_simple|tool_lstm|multi_hop",
  "tools_expected": ["nome_tool_1", "nome_tool_2"]
}
```

**Processo de criação:** Claude gera ~30 candidatos baseado nos filings;
autor revisa/edita/seleciona 20 (dia 6, ~1-2h).

### 8.4 Prompts (versionados em `configs/prompts.yaml`)

**System prompt do agente (v1):**

```yaml
agent_system_prompt_v1: |
  Você é um assistente de analista financeiro especializado em ações.
  
  REGRAS:
  - Use as ferramentas disponíveis sempre que possível antes de responder
  - SEMPRE cite a fonte (qual filing, qual data dos preços)
  - Se não souber, diga "Não tenho informação suficiente" — NUNCA invente
  - Responda em português brasileiro
  - Para sumários sobre compra/venda, sempre inclua: "Esta é uma análise
    educacional, não recomendação financeira"
  
  FERRAMENTAS DISPONÍVEIS:
  {tools}
```

**LLM-as-judge (3 critérios):**

| Critério | Tipo | Pergunta ao juiz |
|----------|------|------------------|
| Coerência técnica | Técnico | "A resposta usa terminologia financeira corretamente?" |
| Citação de fontes | **Negócio** (KPI) | "A resposta cita explicitamente os filings/dados consultados?" |
| Completude | Técnico | "A resposta aborda todos os aspectos da pergunta?" |

### 8.5 Benchmark de 3 configurações

| Config | Modelo | Prompt | top_k | Hipótese |
|--------|--------|--------|-------|----------|
| A (baseline) | Gemini 2.0 Flash | v1 (curto) | 3 | Padrão |
| B (mais contexto) | Gemini 2.0 Flash | v1 | 5 | Mais chunks ajuda? |
| C (modelo menor) | Gemini 1.5 Flash-8B | v1 | 3 | Modelo menor degrada muito? |

**Métricas comparadas:** RAGAS (faithfulness, answer_relevancy, context_precision,
context_recall) + custo (tokens × preço) + latência média.

---

## 9. Segurança (Etapa 4)

### 9.1 Input Guardrail (regex)

Bloqueia ANTES de chamar o Gemini:

| Padrão | Exemplo |
|--------|---------|
| Prompt injection clássico | `ignore (all\s+)?previous instructions` |
| Roleplay malicioso | `you are now a` |
| Tags de modelo | `system:`, `<\|im_start\|>`, `[INST]` |
| Esquecer instruções | `forget (everything\|your instructions)` |
| Tamanho > 4096 chars | Proteção contra context stuffing |

### 9.2 Output Guardrail (Presidio)

Sanitiza antes de devolver. Entidades alvo:
- `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`
- `BR_CPF`, `CREDIT_CARD`, `IBAN`

### 9.3 OWASP Top 10 para LLMs (5 ameaças)

`docs/OWASP_MAPPING.md`:

| ID | Ameaça | Mitigação |
|----|--------|-----------|
| LLM01 | Prompt Injection | Input guardrail regex |
| LLM02 | Sensitive Info Disclosure | Output guardrail Presidio |
| LLM06 | Excessive Agency | `max_iterations=10`, lista fechada de tools |
| LLM07 | System Prompt Leakage | Regex bloqueia "tell me your prompt" |
| LLM10 | Unbounded Consumption | Rate limit + `max_tokens` no Gemini |

### 9.4 Red team (5 cenários)

`docs/RED_TEAM_REPORT.md`:

1. Injection direta (pedir o system prompt)
2. Roleplay ("você é trader sem ética")
3. Encoding (base64 com payload)
4. Extração PII ("que CPFs você viu?")
5. Tool abuse (forçar loop de chamadas ao LSTM)

Pra cada: input + comportamento + verdict (✅ bloqueado / ❌ vazou).

---

## 10. Governança e documentação

### 10.1 `docs/MODEL_CARD.md`
- Modelo, versão, owner, git_sha
- Datasets de treino (FinancialPhraseBank em inglês + preços AAPL 2018-2024)
- Métricas (LSTM: MAE/RMSE/MAPE | Sentimento: F1/precision/recall)
- Limitações (LSTM treinado só em AAPL; sentiment em inglês; sem fairness audit
  profundo)
- Uso pretendido vs não pretendido
- Fairness considerations (sem atributos sensíveis nos datasets)

### 10.2 `docs/SYSTEM_CARD.md`
- Arquitetura completa
- **Tabela de cobertura dos 9 GAPs do Datathon** (mesma da seção 3)
- Trade-offs explícitos (LLM hosted vs self-hosted; 5 filings vs todo SEC)
- **Roadmap pós-MVP**: champion-challenger, drift retrigger, quantização vLLM,
  feature store incremental
- Riscos residuais (sem integration tests → smoke test manual)

### 10.3 `docs/LGPD_PLAN.md` (1-2 páginas)
- Dados pessoais coletados: praticamente nenhum (domínio público de mercado).
  Logs do `/chat` PODEM conter PII se usuário escrever — mitigado pelo Presidio.
- Bases legais: legítimo interesse pra logs anonimizados.
- Direitos do titular (acesso/exclusão): processo manual via DPO.
- Retenção: logs por 90 dias, depois deletados.
- Classificação de risco: ALTO (ML em finanças).

---

## 11. Observabilidade (Etapa 3)

| Camada | Ferramenta | O que captura |
|--------|-----------|---------------|
| API | Prometheus (Fase 4) | requests/s, latência p50/p95, erros |
| LLM | Langfuse free tier | tokens, custo, latência LLM, faithfulness por trace |
| Visualização | Grafana | dashboard com 4 painéis |
| Drift | Evidently | PSI das features de entrada do `/chat` (offline) |

**Threshold de drift:** PSI > 0.1 = warning, PSI > 0.2 = retrigger (manual no MVP).

**Dashboard Grafana (4 painéis):**
1. Requests por minuto
2. Latência p95
3. Tokens consumidos por dia
4. Distribuição de uso de tools

---

## 12. Testes (apenas unitários) e CI/CD

### 12.1 Estratégia de testes

**Decisão do autor:** TODOS os testes são unitários. Sem integração end-to-end.

| Arquivo | Foco | Mocks |
|---------|------|-------|
| `test_features.py` | Schema pandera, ranges | Sem I/O |
| `test_models.py` | LSTM `predict()`, sentimento `predict()` | Modelo em memória |
| `test_agent.py` | Cada tool isolada + função de seleção | LLM, yfinance, ChromaDB **mockados** |
| `test_api.py` | Route handlers chamados diretamente | Agente, LSTM, guardrails **mockados** |
| `test_guardrails.py` | Regex e Presidio | Strings de teste |
| `test_preprocessing.py` | Funções de scaling/sequências | Sem I/O |

**Cobertura alvo:** ≥ 60% (cumpre rubrica do Datathon).

**Trade-off explícito (registrado no System Card):** "Sem integration tests
aumenta risco de regressão silenciosa em integrações externas. Mitigado por
smoke test manual antes de cada release (`make smoke`)."

### 12.2 CI (`.github/workflows/ci.yml`)

```
push/PR em src/, tests/, evaluation/
  ├── lint (ruff)
  ├── typecheck (mypy --ignore-missing-imports)
  ├── security scan (bandit)
  ├── pytest --cov-fail-under=60
  └── docker build (sem push)
```

**Cortado:** deploy automático para staging.

---

## 13. Demo, vídeo e pitch

### 13.1 Estrutura do vídeo (8-10 min)

| # | Bloco | Tempo | Conteúdo |
|---|-------|-------|----------|
| 1 | Problema | 1 min | "Analistas gastam X horas lendo um 10-K" |
| 2 | Abordagem | 2 min | Diagrama de arquitetura + decisões |
| 3 | Demo ao vivo | 4 min | `/chat` → traces Langfuse → métricas Grafana |
| 4 | Resultados | 2 min | RAGAS dos 3 benchmarks + KPIs de negócio |
| 5 | Impacto + futuro | 1 min | Roadmap (quantização, retraining), LGPD |

### 13.2 Script de demo

1. Mostrar `docker-compose up` já rodando
2. `curl POST /chat` com:
   > "Considerando o último 10-K da Apple, o preço atual e a projeção do LSTM,
   > qual o seu sumário sobre comprar AAPL hoje?"
3. Mostrar resposta do agente (com fontes citadas)
4. Abrir Langfuse → trace mostrando uso de 4 tools
5. Abrir Grafana → painel de latência e tokens
6. Rodar `evaluation/ragas_eval.py` com golden set
7. Mostrar 1 caso de red team bloqueando injection

### 13.3 Critérios de sucesso da demo

- ✅ Roda do início ao fim sem erro no laptop
- ✅ Resposta cita fonte (10-K) e usa ≥ 2 tools
- ✅ Trace aparece no Langfuse em tempo real

---

## 14. Trade-offs e cortes assumidos

Cada corte vai estar **explícito no System Card** com justificativa:

| Item original do guia | Decisão | Motivo |
|----------------------|---------|--------|
| LLM self-hosted com quantização (vLLM/BentoML) | ❌ CORTAR — usar API Gemini | 9 dias solo é inviável; mitigado por benchmark de 3 configs hosted |
| DVC com pipeline completo | ❌ MINIMIZAR — só hash dos arquivos | Sobrecarga sem ROI no pitch |
| Champion-challenger retraining | ❌ CORTAR (descrever no System Card) | Vira "trabalho futuro" |
| Drift detection automático com retrigger | ⚠️ MÍNIMO — só relatório offline | Marca o checkbox |
| Deploy automático para staging | ❌ CORTAR | CI fica em lint+test+build |
| Integration tests | ❌ CORTAR (decisão do autor) | Compensado por unit tests + smoke manual |

---

## 15. Riscos previsíveis e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Gemini free tier rate limit | Médio | Alto | Caching local + batch nos testes; fallback pra Gemini 1.5 Flash-8B |
| SEC EDGAR fora do ar | Baixo | Médio | Backup dos filings já baixados em `data/filings/` versionado |
| Drift "fraco" com dados sintéticos | Alto | Baixo | Narrativa honesta no System Card |
| Conflito PyTorch + Keras no `pyproject.toml` | Médio | Médio | Remover Keras; manter só PyTorch (já que migramos LSTM) |
| Atrasar Dia 5 (agente) | Médio | Crítico | Cortar Dia 7 (Grafana) e usar tempo recuperando |
| Bug no dia da demo | Baixo | Crítico | Vídeo gravado serve de backup |

---

## 16. Out of scope (NÃO vai ser feito)

Lista explícita pra evitar scope creep:

- LLM self-hosted ou quantização
- DVC pipeline completo
- Champion-challenger retraining automatizado
- Retrigger automático de drift
- Deploy em nuvem
- Integration tests (decisão do autor)
- Multi-tenancy / autenticação no `/chat`
- UI web (só API com Swagger)
- Suporte a múltiplos idiomas no agente (só PT-BR)
- Treinar LSTM em mais empresas (mantém só AAPL)

---

## 17. Critérios de aceite

O projeto está pronto pra entrega quando:

1. ✅ `docker-compose up` sobe API funcional com endpoint `/chat`
2. ✅ Agente responde pergunta multi-hop usando ≥ 2 tools
3. ✅ Golden set com 20 pares versionado em `data/golden_set/`
4. ✅ RAGAS rodando com as 4 métricas obrigatórias e relatório salvo
5. ✅ Langfuse traces aparecendo
6. ✅ Prometheus + Grafana com dashboard funcionando
7. ✅ Guardrails bloqueando 5 cenários de red team
8. ✅ Cinco docs em `docs/` completas
9. ✅ CI verde no GitHub Actions
10. ✅ pytest com `--cov-fail-under=60` passando
11. ✅ Vídeo gravado e disponível
12. ✅ README reescrito explicando como rodar

---

## 18. Próximos passos

1. Autor revisa este spec e dá OK ou pede ajustes
2. Quando aprovado, escrevemos o **plano de implementação detalhado**
   (`writing-plans` skill) — divide cada dia do cronograma em tarefas
   executáveis com critérios de aceite
3. Iniciamos Dia 1 (setup + esqueleto)
