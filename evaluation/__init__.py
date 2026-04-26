"""Pacote de avaliacao do agente financeiro.

Contem:
- generate_golden_candidates: gera candidatos para o golden set.
- ragas_eval: avaliacao com 4 metricas RAGAS (faithfulness, answer_relevancy,
  context_precision, context_recall).
- llm_judge: LLM-as-judge com 3 criterios (incluindo citacao de fontes,
  KPI de negocio).
- benchmark_configs: benchmark de 3 configuracoes do agente.
"""
