"""Output guardrail: redacao de PII com Presidio.

Roda na saida do agente (`/chat`) antes de devolver ao cliente. Detecta:
- PERSON (nomes proprios)
- EMAIL_ADDRESS
- PHONE_NUMBER
- CREDIT_CARD
- IBAN_CODE

Estrategia de idioma:
1. Tenta `language="pt"` primeiro (modelo `pt_core_news_sm` ja instalado).
2. Se falhar (ex: presidio nao tem o NLP engine pra PT), faz fallback `en`.

Por que nao bloquear? Sanitizar e menos disruptivo que bloquear: o usuario ainda
recebe a resposta, apenas com PII redacionado em tags `<EMAIL_ADDRESS>` etc.

Notas LGPD: ver `docs/LGPD_PLAN.md` Secao 3 (anonimizacao automatica como
direito do titular - Art. 18, IV).
"""

from __future__ import annotations

import logging

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# Engines do Presidio sao caros pra construir (carregam spaCy). Mantemos
# instancias modulo-globais — thread-safe pra leitura, que e o nosso caso.
_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# Entidades alvo. Note: CPF nao e nativo do Presidio mas EMAIL/PHONE casam o
# essencial pro nosso contexto + PERSON cobre nomes.
ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
]


def sanitize_output(text: str, language: str = "pt") -> str:
    """Anonimiza PII no texto.

    Args:
        text: texto bruto a ser sanitizado.
        language: codigo ISO do idioma. Default `pt`. Faz fallback pra `en` se
            o pipeline de PT nao estiver disponivel.

    Returns:
        Texto com PII substituido por placeholders `<EMAIL_ADDRESS>` etc.
        Se nada for detectado, retorna `text` inalterado (zero-cost path).
    """
    try:
        results = _analyzer.analyze(text=text, language=language, entities=ENTITIES)
    except Exception:  # noqa: BLE001 — fallback defensivo
        # Pipeline de PT pode nao estar registrado. Tentamos EN.
        logger.debug("Analyzer falhou em '%s', fallback para 'en'", language)
        results = _analyzer.analyze(text=text, language="en", entities=ENTITIES)

    if not results:
        return text

    logger.warning("PII detectado: %d entidade(s)", len(results))
    sanitized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return sanitized.text
