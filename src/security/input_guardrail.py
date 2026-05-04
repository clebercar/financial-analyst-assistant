"""Input guardrail anti-prompt-injection.

Aplica duas camadas defensivas no input do `/chat`:

1. **Tamanho maximo** (`MAX_LENGTH = 4096`): limita custo + reduz superficie de ataque.
2. **Padroes regex** (`INJECTION_PATTERNS`): detecta tentativas comuns de prompt
   injection (jailbreaks, role override, leak do system prompt). E uma defesa
   *barata* e *rapida* — nao substitui guardrails baseados em LLM (ex: Llama Guard).

Limitacoes conhecidas (documentadas em `docs/RED_TEAM_REPORT.md`):
- nao decodifica base64/hex (cenario 3 do red team passa);
- nao cobre injection multi-turno;
- regex em ingles + portugues basico — pode falhar em portugues sofisticado.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 7 padroes de prompt injection. Mantidos em ordem do mais comum para o mais raro
# para curto-circuitar rapido na maioria dos casos.
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(a|an)\b",
    r"system\s*:",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"forget\s+(everything|all|your\s+instructions)",
    r"reveal\s+(the\s+)?(system|hidden)\s+prompt",
]

# Compilamos uma vez no import — `re.compile` e caro pra fazer por request.
_compiled: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# 4096 chars cobre perguntas elaboradas sem permitir injection de payloads
# inteiros. Alinhado com `ChatRequest.pergunta` em `src/serving/schemas.py`.
MAX_LENGTH: int = 4096


def validate_input(text: str) -> tuple[bool, str]:
    """Valida o input do usuario.

    Args:
        text: texto bruto vindo do usuario (campo `pergunta` do `/chat`).

    Returns:
        Tupla `(is_valid, reason)`:
        - `(True, "OK")` se passou.
        - `(False, "Input bloqueado: ...")` se algum pattern bateu ou texto longo.
    """
    if len(text) > MAX_LENGTH:
        logger.warning("Input excede MAX_LENGTH (%d): %d chars", MAX_LENGTH, len(text))
        return False, f"Input bloqueado: excede {MAX_LENGTH} caracteres"

    for pat in _compiled:
        if pat.search(text):
            # Nunca logamos o input completo — pode ter PII. Truncamos em 120 chars.
            logger.warning("Prompt injection detectado: %s", text[:120])
            return False, "Input bloqueado: padrao suspeito detectado"

    return True, "OK"
