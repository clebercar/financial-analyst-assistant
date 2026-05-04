# OWASP Top 10 para LLM Applications — Mapeamento

Referencia: OWASP (2025) — https://owasp.org/www-project-top-10-for-large-language-model-applications/

Mapeia as ameacas relevantes ao assistente e a mitigacao implementada (ou
planejada como roadmap).

## Tabela de mapeamento

| ID    | Ameaca                            | Risco no sistema                                                                       | Mitigacao implementada                                                                                       | Status                                              |
|-------|-----------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| LLM01 | Prompt Injection                  | Usuario pode tentar manipular o agente (jailbreak, role override, leak de prompt)      | `src/security/input_guardrail.py` — 7 padroes regex (case-insensitive) + limite 4096 chars                   | Implementado                                        |
| LLM02 | Sensitive Information Disclosure  | Logs ou outputs podem vazar PII (CPF, email, telefone, nome de cliente)                | `src/security/output_guardrail.py` — Presidio + spaCy PT/EN; entidades: PERSON, EMAIL, PHONE, CC, IBAN       | Implementado                                        |
| LLM06 | Excessive Agency                  | Agente poderia entrar em loop de tool calls e queimar custo Gemini / fontes externas   | `max_iterations=10` no `AgentExecutor` + lista fechada de 4 tools (sem tool de codigo arbitrario)            | Implementado                                        |
| LLM07 | System Prompt Leakage             | Usuario pode tentar extrair o prompt do sistema com perguntas adversariais             | Regex `reveal\s+(the\s+)?(system\|hidden)\s+prompt` no input guardrail; system prompt nao e logado em traces | Implementado                                        |
| LLM10 | Unbounded Consumption             | Custo Gemini pode escalar com requests massivos; sem rate limit por IP                 | `max_tokens` configurado no Gemini + limite de 4096 chars no input. Falta `slowapi` para rate limit por IP   | Parcial — rate limit por IP fica no roadmap         |

## Ameacas nao cobertas neste momento

| ID    | Ameaca                       | Por que nao cobrimos                                                              | Plano                                                                  |
|-------|------------------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------|
| LLM03 | Supply Chain                 | Auditoria de pesos de modelos terceiros (Gemini, embeddings) e custosa            | Pinning de versoes + scan automatico via `pip-audit`                   |
| LLM04 | Data Poisoning               | RAG indexa apenas filings publicos da SEC (fonte autoritaria)                     | Hash de cada doc indexado + re-verificacao periodica                   |
| LLM05 | Improper Output Handling     | Output e texto puro mostrado ao usuario; nao executamos como codigo               | Continua suficiente                                                    |
| LLM08 | Vector Weaknesses            | ChromaDB local; risco baixo                                                       | Rodar `prompt-injection-bench` contra a colecao                        |

## Roadmap

1. **Rate limit por IP** com `slowapi` no `/chat` (cobre LLM10 totalmente).
2. **Decodificacao base64/hex antes do regex** (cobre Cenario 3 do Red Team).
3. **Llama Guard local** como segundo filtro de input (cobre LLM01 com modelo,
   nao so regex).
4. **Auditoria de embeddings** com seed inputs conhecidos para detectar drift
   semantico no ChromaDB (cobre LLM08).

## Referencias cruzadas

- Cenarios adversariais executados: `docs/RED_TEAM_REPORT.md`
- Conformidade LGPD (relacionada a LLM02): `docs/LGPD_PLAN.md`
- Decisoes de design e trade-offs: `docs/SYSTEM_CARD.md`
