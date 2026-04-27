# Model Card - Datathon Fase 5

Esse Model Card documenta os dois modelos de ML classico do projeto.
Metricas reais foram extraidas de `mlruns/` (MLflow).

---

## 1. LSTM - Previsao de Preco de AAPL

### Identificacao
- **Nome:** lstm_aapl
- **Versao:** 0.1.0
- **Owner:** Cleber Carvalho (contatoclebercarvalho@gmail.com)
- **Framework:** PyTorch 2.x
- **Tipo:** regressao (series temporais)
- **MLflow run:** `mlruns/356059821066200539/c950b245bdfc47e79b23df0d5886301c`

### Dados de Treinamento
- **Fonte:** Yahoo Finance via `yfinance`
- **Ticker:** AAPL
- **Periodo:** 2018-01-01 a 2024-12-31 (~7 anos de pregoes)
- **Pre-processamento:** `MinMaxScaler` para o intervalo [0, 1], janela
  deslizante de 60 dias (input) -> 1 dia (target)
- **Split:** 80% treino / 20% teste (sem embaralhar - serie temporal)

### Arquitetura
- `LSTM(50)` -> `Dropout(0.2)` -> `LSTM(50)` -> `Dropout(0.2)` -> `Dense(25, ReLU)` -> `Dense(1)`
- Otimizador: Adam (lr=0.001) | Loss: MSE | Epochs: 50 | Batch: 32

### Metricas (test set - reais do MLflow)

| Metrica | Valor    | Interpretacao                               |
|---------|----------|---------------------------------------------|
| MAE     | 9.4481   | Erramos em media US$9.45 no preco           |
| RMSE    | 12.3964  | Penaliza erros grandes; ainda razoavel      |
| MAPE    | 4.39%    | Erro percentual medio - bom para baseline   |

### Limitacoes Conhecidas
- Treinado **apenas** em AAPL - generalizacao para outros tickers e
  experimental (a tool `prever_preco_lstm` retorna `aviso` quando ticker != AAPL).
- Nao considera noticias, earnings, eventos macro - apenas historico de
  precos de fechamento.
- Mercado de acoes e fundamentalmente imprevisivel; modelo e educacional.
- Treinado ate 2024-12-31; performance em regime pos-treino degrada com
  drift (ver `src/monitoring/drift_report.py`).

### Uso Pretendido
- Educacao financeira, demonstracao de tecnicas de ML em series temporais.
- Tool auxiliar para analise por humanos.
- **NAO recomendado** para decisoes automatizadas de compra/venda.
- **NAO recomendado** como aconselhamento financeiro.

### Fairness
- Dataset nao tem atributos sensiveis (sem CPF, genero, etc.) - preco de
  mercado e dado publico agregado. `fairness_checked=true` registrado no
  MLflow run.

---

## 2. Sentimento Financeiro - TF-IDF + LogReg

### Identificacao
- **Nome:** sentiment_phrasebank
- **Versao:** 0.1.0
- **Owner:** Cleber Carvalho
- **Framework:** scikit-learn
- **Tipo:** classificacao multiclasse (3 classes: positive / negative / neutral)
- **MLflow run:** `mlruns/913574969878309190/cfcf2e6ea642436e9ea7ba6545fcc559`

### Dados de Treinamento
- **Fonte:** Hugging Face dataset `financial_phrasebank`
- **Subset:** `sentences_75agree` (~75% de concordancia entre anotadores)
- **Idioma:** ingles
- **Split:** 80% treino / 20% teste, estratificado pela classe

### Arquitetura
- TF-IDF (n-gram 1-2, `max_features=5000`, `min_df=2`) -> Logistic Regression
  (`class_weight=balanced`, sem regularizacao customizada).

### Metricas (test set - reais do MLflow)

| Metrica         | Valor   | Interpretacao                                      |
|-----------------|---------|----------------------------------------------------|
| F1 macro        | 0.8044  | Boa performance considerando 3 classes             |
| Precision macro | 0.8139  | Pouca falsa-classificacao positiva                 |
| Recall macro    | 0.7957  | Cobertura razoavel das 3 classes                   |

### Limitacoes
- Treinado em **ingles** - performance em portugues e degradada. O agente
  passa apenas trechos em ingles para essa tool (10-K filings da SEC sao
  em ingles).
- Dominio: noticias e relatorios formais; texto informal/redes sociais nao
  foi testado.
- 3 classes apenas (positive/negative/neutral) - nuances como "cautiously
  optimistic" caem no baco neutro.

### Fairness
- Dataset sem atributos sensiveis. `fairness_checked=true` registrado no
  MLflow run.

---

## Governanca

Ambos os modelos sao registrados no MLflow com schema obrigatorio de tags:
- `owner`, `model_name`, `model_version`, `model_type`, `phase`,
  `risk_level`, `fairness_checked`, `git_sha`, `training_data_version`.

Isso atende ao GAP 05 do Datathon (versionamento + governanca de modelos).

## Re-treinamento

- **Atual:** retraining manual via `make train-lstm` / `make train-sentiment`.
- **Roadmap:** champion-challenger com aprovacao humana antes de promover -
  ver `docs/SYSTEM_CARD.md` Secao 4.3.
