# Schemas Pydantic que definem o formato dos dados que a API aceita e retorna.
# Isso garante validacao automatica: se o usuario mandar dados fora do formato,
# a API ja retorna um erro explicando o que ta errado.


from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str


class ChatRequest(BaseModel):
    """Pergunta enviada ao agente financeiro."""

    pergunta: str = Field(
        ...,
        max_length=4096,
        description="Pergunta em linguagem natural sobre acoes, filings ou sentimento.",
    )


class ChatResponse(BaseModel):
    """Resposta do agente, com metadados de execucao para observabilidade."""

    resposta: str = Field(..., description="Resposta final do agente em portugues.")
    iteracoes: int = Field(
        ..., description="Quantos passos (Action+Observation) o agente executou."
    )
    tools_chamadas: list[str] = Field(
        default_factory=list, description="Nome das tools chamadas, em ordem."
    )
