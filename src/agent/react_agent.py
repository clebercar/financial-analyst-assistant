"""Agente ReAct usando LangChain 1.x + Gemini com 4 tools financeiras.

Notas sobre versoes (LangChain 1.x vs 0.2.x):
- A partir do LangChain 1.x, `create_react_agent` e `AgentExecutor` foram
  removidos do `langchain.agents`. A API equivalente fica em
  `langgraph.prebuilt.create_react_agent`, que retorna um `CompiledStateGraph`
  ao inves de um `AgentExecutor`. Para preservar a interface esperada pelo
  endpoint /chat (`agent.invoke({"input": ...})` retornando
  `{"output", "intermediate_steps"}`), definimos aqui uma classe
  `AgentExecutor` adapter que envolve o grafo e expoe esse contrato.
- `StructuredTool` veio do `langchain_core.tools` (langchain.tools nao expoe
  mais o simbolo em 1.x).

A funcao `create_financial_agent` e o ponto de entrada usado pelo endpoint
/chat. Aceita um `model_name` opcional, util pro benchmark do Dia 6 onde
trocamos entre Gemini variantes (2.0 flash vs 1.5 pro etc).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.agent import tools as t

load_dotenv()
logger = logging.getLogger(__name__)


class AgentExecutor:
    """Adapter sobre o `CompiledStateGraph` do langgraph.

    Expõe o mesmo contrato `invoke({"input": pergunta}) ->
    {"output": str, "intermediate_steps": list[(action, observation)]}` que o
    `AgentExecutor` legado do langchain 0.x oferecia. Mantém o codigo do /chat
    desacoplado da API interna do langgraph.

    `intermediate_steps` aqui vira lista de tuplas (FakeAction, observation),
    onde `FakeAction.tool` é o nome da tool. O endpoint /chat extrai apenas
    `step[0].tool` — esse contrato é o que importamos preservar.
    """

    class _Action:
        """Mini wrapper só pra ter `.tool` (o /chat só lê esse atributo)."""

        def __init__(self, tool: str, tool_input: Any) -> None:
            self.tool = tool
            self.tool_input = tool_input

    def __init__(
        self,
        agent: Any,
        tools: list[StructuredTool],
        max_iterations: int = 10,
        verbose: bool = True,
        handle_parsing_errors: bool = True,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.agent = agent
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.handle_parsing_errors = handle_parsing_errors
        # Lista de callbacks (ex: Langfuse CallbackHandler) repassados no
        # invoke do langgraph via RunnableConfig.
        self.callbacks: list[Any] = callbacks or []

    def invoke(self, payload: dict) -> dict:
        """Roda o agente sobre `payload['input']` e devolve dict legado."""
        pergunta = payload["input"]
        config: dict[str, Any] = {"recursion_limit": self.max_iterations * 2}
        if self.callbacks:
            # Langgraph (1.x) aceita `callbacks` no RunnableConfig — eles
            # sao injetados em todas as Runnable internas (LLM + tools).
            config["callbacks"] = self.callbacks
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=pergunta)]},
            config=config,
        )
        messages = result.get("messages", [])

        # Extrai resposta final: ultima AIMessage sem tool_calls
        output = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                output = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Reconstroi intermediate_steps a partir de tool_calls / ToolMessage
        intermediate_steps: list[tuple[Any, Any]] = []
        pending: dict[str, AgentExecutor._Action] = {}
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in getattr(msg, "tool_calls", []) or []:
                    raw_name = (
                        call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                    )
                    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                    call_id = (
                        call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                    )
                    action = AgentExecutor._Action(tool=str(raw_name or ""), tool_input=args)
                    if call_id:
                        pending[call_id] = action
                    else:
                        intermediate_steps.append((action, None))
            elif isinstance(msg, ToolMessage):
                call_id = getattr(msg, "tool_call_id", None)
                resolved: AgentExecutor._Action | None = (
                    pending.pop(call_id) if call_id and call_id in pending else None
                )
                if resolved is None:
                    resolved = AgentExecutor._Action(
                        tool=str(getattr(msg, "name", "unknown")), tool_input=None
                    )
                intermediate_steps.append((resolved, msg.content))
        # Adiciona pending sem ToolMessage correspondente (deveria ser raro)
        for action in pending.values():
            intermediate_steps.append((action, None))

        return {"output": output, "intermediate_steps": intermediate_steps}


def _build_tools() -> list[StructuredTool]:
    """Constroi a lista de tools que o agente pode chamar."""
    return [
        StructuredTool.from_function(
            func=t.consultar_preco,
            name="consultar_preco",
            description=(
                "Consulta preco atual e variacao dos ultimos 30 dias de uma acao. "
                "Input: ticker (string)."
            ),
        ),
        StructuredTool.from_function(
            func=t.prever_preco_lstm,
            name="prever_preco_lstm",
            description=(
                "Preve preco de fechamento para os proximos N dias uteis usando LSTM. "
                "Input: ticker (string), dias (int, default 5)."
            ),
        ),
        StructuredTool.from_function(
            func=t.analisar_sentimento,
            name="analisar_sentimento",
            description=(
                "Classifica sentimento de texto financeiro "
                "(positive/neutral/negative). Input: texto (string)."
            ),
        ),
        StructuredTool.from_function(
            func=t.buscar_em_filings,
            name="buscar_em_filings",
            description=(
                "Busca trechos relevantes em filings 10-K e 10-Q da SEC. "
                "Input: query (string), ticker opcional, top_k (int)."
            ),
        ),
    ]


def create_financial_agent(
    model_name: str | None = None,
    callbacks: list[Any] | None = None,
) -> AgentExecutor:
    """Cria agente ReAct configurado para analise financeira.

    Args:
        model_name: nome do modelo Gemini (override do `agent.llm_model` do
            `configs/model_config.yaml`). Util pro benchmark do Dia 6.
        callbacks: lista opcional de callbacks (ex: Langfuse) que serao
            repassados via RunnableConfig em cada invoke. Permite tracing
            transparente sem acoplar o agente ao Langfuse.

    Returns:
        AgentExecutor adapter pronto pra invoke({"input": pergunta}).
    """
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)["agent"]
    with open("configs/prompts.yaml") as f:
        prompts = yaml.safe_load(f)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao definida")

    llm = ChatGoogleGenerativeAI(
        model=model_name or cfg["llm_model"],
        temperature=cfg["temperature"],
        google_api_key=api_key,
    )

    # System prompt: usamos o template `agent_system_v1` mas extraimos so a
    # parte de instrucoes (tools/scratchpad sao tratados pelo langgraph).
    system_prompt = prompts["agent_system_v1"]

    tools = _build_tools()
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=cfg["max_iterations"],
        handle_parsing_errors=True,
        callbacks=callbacks,
    )
