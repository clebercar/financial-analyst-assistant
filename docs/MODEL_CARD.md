# Model Card

Documentacao dos dois modelos de ML classico utilizados pelo agente.
Metricas extraidas de runs reais do MLflow (`mlruns/`).

---

## 1. LSTM — Previsao de Preco de AAPL

### Identificacao
- **Nome:** `lstm_aapl`
- **Versao:** 0.1.0
- **Framework:** PyTorch 2.x
- **Tipo:** regressao (series temporais)
- **Owner:** ml-team

### Dados de Treinamento
- **Fonte:** Yahoo Finance via `yfinance`
- **Ticker:** AAPL
- **Periodo:** 2018-01-01 a 2024-12-31 (~7 anos de pregoes)
- **Pre-processamento:** `MinMaxScaler` para o intervalo [0, 1], janela
  deslizante de 60 dias (input) -> 1 dia (target)
- **Split:** 80% treino / 20% teste (sem embaralhar — serie temporal)

### Arquitetura
- `LSTM(50)` -> `Dropout(0.2)` -> `LSTM(50)` -> `Dropout(0.2)` -> `Dense(25, ReLU)` -> `Dense(1)`
- Otimizador: Adam (lr=0.001) | Loss: MSE | Epochs: 50 | Batch: 32

### Metricas (test set)

| Metrica | Valor    | Interpretacao                               |
|---------|----------|---------------------------------------------|
| MAE     | 9.4481   | Erro absoluto medio em USD                  |
| RMSE    | 12.3964  | Penaliza erros grandes                      |
| MAPE    | 4.39 %   | Erro percentual medio                       |

### Limitacoes Conhecidas
- Treinado **apenas** em AAPL — generalizacao para outros tickers e
  experimental (a tool `prever_preco_lstm` retorna campo `aviso` quando
  ticker != AAPL).
- Nao considera noticias, earnings ou eventos macro — apenas historico de
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
- Dataset nao tem atributos sensiveis (sem CPF, genero, etc.) — preco de
  mercado e dado publico agregado.

---

## 2. Sentimento Financeiro — TF-IDF + Logistic Regression

### Identificacao
- **Nome:** `sentiment_phrasebank`
- **Versao:** 0.1.0
- **Framework:** scikit-learn
- **Tipo:** classificacao multiclasse (3 classes: positive / negative / neutral)
- **Owner:** ml-team

### Dados de Treinamento
- **Fonte:** Hugging Face dataset `financial_phrasebank`
- **Subset:** `sentences_75agree` (~75% de concordancia entre anotadores)
- **Idioma:** ingles
- **Split:** 80% treino / 20% teste, estratificado pela classe

### Arquitetura
- TF-IDF (n-gram 1-2, `max_features=5000`, `min_df=2`) -> Logistic Regression
  (`class_weight=balanced`).

### Metricas (test set)

| Metrica         | Valor   | Interpretacao                                      |
|-----------------|---------|----------------------------------------------------|
| F1 macro        | 0.8044  | Boa performance considerando 3 classes             |
| Precision macro | 0.8139  | Pouca falsa-classificacao positiva                 |
| Recall macro    | 0.7957  | Cobertura razoavel das 3 classes                   |

### Limitacoes
- Treinado em **ingles** — performance em portugues e degradada. O agente
  passa apenas trechos em ingles para essa tool (10-K filings da SEC sao
  em ingles).
- Dominio: noticias e relatorios formais; texto informal/redes sociais nao
  foi testado.
- 3 classes apenas — nuances como "cautiously optimistic" caem no balde
  neutro.

### Fairness
- Dataset sem atributos sensiveis.

---

## Governanca

Ambos os modelos sao registrados no MLflow com schema obrigatorio de tags:
`owner`, `model_name`, `model_version`, `model_type`, `risk_level`,
`fairness_checked`, `git_sha`, `training_data_version`.

## Re-treinamento

- **Atual:** retraining manual via `make train-lstm` / `make train-sentiment`.
- **Roadmap:** champion-challenger com aprovacao humana antes de promover
  (ver `docs/SYSTEM_CARD.md`).
