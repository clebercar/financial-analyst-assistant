"""Wrapper Langfuse para tracing de chamadas LLM.

Langfuse 4.x mudou bastante em relacao ao 2.x:
- O `CallbackHandler` agora vive em `langfuse.langchain` (em vez de
  `langfuse.callback`).
- A SDK 4.x prefere autoconfiguracao via variaveis de ambiente
  (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`). O
  `CallbackHandler` ja le essas vars internamente, entao nao passamos
  parametros pro construtor.
- Se as chaves nao estiverem setadas, o handler instancia mas fica
  silenciosamente desabilitado. Pra evitar overhead e facilitar testes,
  retornamos `None` quando nao ha chaves: o caller (app.py) checa e nao
  registra callback nenhum.

Por que `get_langfuse_callback()` retorna None sem chaves?
- A app precisa funcionar offline (testes, dev sem cadastro Langfuse).
- Devolver None mantem `create_financial_agent(callbacks=None)` funcional.
- Levantar excecao aqui quebraria todo `/chat` em ambiente sem Langfuse.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_langfuse_callback() -> Any | None:
    """Retorna handler Langfuse se chaves disponiveis, senao None.

    Returns:
        Instancia de `langfuse.langchain.CallbackHandler` se
        `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` estiverem definidas;
        caso contrario `None`. Em caso de erro de import ou instanciacao,
        loga warning e retorna None — nunca propaga excecao.
    """
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        logger.info("Langfuse keys ausentes - tracing desabilitado")
        return None

    try:
        # Import local: evita custo de import se o caller nao precisar.
        from langfuse.langchain import CallbackHandler  # type: ignore[import-untyped]

        # Em 4.x o CallbackHandler le LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST das
        # env vars automaticamente. Nao passamos parametros pra deixar a
        # SDK fazer o trabalho dela (e respeitar opcoes como tracing_enabled).
        return CallbackHandler()
    except Exception as e:  # noqa: BLE001 - logar e seguir
        logger.warning("Falha ao instanciar Langfuse CallbackHandler: %s", e)
        return None
