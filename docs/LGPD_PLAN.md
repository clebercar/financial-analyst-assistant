# Plano de Conformidade LGPD

Lei 13.709/2018 (Lei Geral de Protecao de Dados Pessoais).

## 1. Mapeamento de Dados Pessoais

### 1.1 Dados coletados pelo sistema

- **Endpoint `/chat`:** input do usuario (texto livre).
  - Pode conter PII se o usuario escolher escrever (ex: "meu CPF e X").
  - **Tratamento:** logs de input sao truncados em 120 chars (ver
    `src/security/input_guardrail.py`); output e sanitizado pelo Presidio
    antes de devolver/persistir (`src/security/output_guardrail.py`).
- **Logs operacionais:** IP do cliente HTTP (Prometheus default).
  - **Base legal:** legitimo interesse (seguranca operacional).
- **Tracing Langfuse:** input + output armazenados em servidor externo (UE).
  - **Base legal:** legitimo interesse + consentimento implicito.

### 1.2 Dados que NAO sao coletados
- CPF, RG, identificacao pessoal direta.
- Dados financeiros pessoais (saldo, posicao em acoes reais do usuario).
- Localizacao precisa.
- Cookies de tracking.

## 2. Bases Legais

| Operacao                          | Base legal LGPD                              |
|-----------------------------------|----------------------------------------------|
| Logs operacionais (IP)            | Legitimo interesse (Art. 7 IX)               |
| Tracing Langfuse                  | Legitimo interesse + consentimento implicito |
| Persistencia de input do `/chat`  | Legitimo interesse para melhoria do modelo   |

## 3. Direitos do Titular

| Direito                            | Como atendemos                                                             |
|------------------------------------|----------------------------------------------------------------------------|
| Acesso (Art. 18 I)                 | Titular envia email para o DPO solicitando dump dos logs com seu IP        |
| Anonimizacao (Art. 18 IV)          | Presidio aplicado automaticamente em outputs do `/chat`                    |
| Eliminacao (Art. 18 VI)            | DPO executa pipeline de delecao por IP                                     |
| Portabilidade (Art. 18 V)          | Mesmos dumps em JSON                                                       |
| Revogacao de consentimento         | Usuario simplesmente para de usar — nao mantemos cadastro persistente      |

## 4. Retencao

- **Logs Prometheus:** 30 dias (configurado em `prometheus.yml`).
- **Traces Langfuse:** 90 dias (free tier default).
- **Inputs persistidos para retraining:** 90 dias, sanitizados em pipeline batch.

## 5. Responsabilidades

- **Controlador:** ml-team
- **DPO/Encarregado:** ml-team
- **Operadores externos:** ver Secao 6.

## 6. Operadores Externos (Subcontratados)

Terceiros que processam dados em nosso nome. Cada um tem DPA padrao publico
do fornecedor.

| Operador                    | Dados transferidos                  | Localizacao | Salvaguardas                                     |
|-----------------------------|-------------------------------------|-------------|--------------------------------------------------|
| Google (Gemini API)         | Input + output do agente            | EUA         | DPA padrao da Google AI; SCC para transferencia  |
| Langfuse Cloud              | Input + output (traces LLM)         | UE          | DPA conforme GDPR                                |
| Yahoo Finance (yfinance)    | Apenas tickers consultados          | EUA         | Dados publicos, sem PII                          |
| SEC EDGAR                   | Apenas CIK / numero de filing       | EUA         | Dados publicos                                   |

**Nota:** o input do usuario passa pelo Gemini (operador externo) e pelo
Langfuse (operador externo). Isso e divulgado no README e no System Card.

## 7. Classificacao de Risco

**Risco: ALTO**

Justificativa:
- Dominio financeiro (regulado pela CVM, BACEN).
- Possibilidade de vazamento de PII via logs (mitigado mas nao eliminado:
  o Presidio nao e 100% preciso).
- Decisoes com impacto financeiro potencial — apesar do disclaimer.

## 8. Plano de Resposta a Incidentes

1. **Deteccao:** alerta no Prometheus quando rate de erro > 1% ou quando
   o contador `chat_requests_total{status="blocked_input"}` cresce de forma
   anomala.
2. **Contencao:** desligar `/chat` (`docker-compose stop api`).
3. **Avaliacao:** revisar logs (ja sanitizados pelo Presidio na saida; logs
   de input truncados em 120 chars).
4. **Notificacao:** se PII vazou de fato, notificar ANPD em 72 h conforme
   Art. 48 e titulares afetados conforme Art. 48 § 1.
5. **Remediacao:** patch + retraining se houver suspeita de data poisoning.

## 9. Atualizacoes deste plano

Este plano e revisto a cada nova versao maior do sistema (mudanca de
modelo, novo operador externo, novo endpoint que coleta dado novo).
