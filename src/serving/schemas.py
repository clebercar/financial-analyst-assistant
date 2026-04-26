# Schemas Pydantic que definem o formato dos dados que a API aceita e retorna.
# Isso garante validacao automatica: se o usuario mandar dados fora do formato,
# a API ja retorna um erro explicando o que ta errado.


from pydantic import BaseModel, Field


class PrevisaoRequest(BaseModel):
    """
    Corpo da requisicao para o endpoint de previsao.
    O usuario precisa mandar uma lista com os precos de fechamento dos ultimos N dias.
    """

    precos_fechamento: list[float] = Field(
        ...,
        min_length=60,
        description="Lista com precos de fechamento dos ultimos dias (minimo 60 valores)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "precos_fechamento": [
                        266.0,
                        271.24,
                        275.66,
                        276.71,
                        277.29,
                        278.59,
                        282.84,
                        285.92,
                        283.88,
                        280.44,
                        278.52,
                        277.63,
                        276.92,
                        278.52,
                        277.77,
                        278.02,
                        273.85,
                        274.35,
                        271.59,
                        271.94,
                        273.41,
                        270.72,
                        272.11,
                        273.55,
                        273.14,
                        273.5,
                        272.82,
                        271.61,
                        270.76,
                        267.01,
                        262.11,
                        260.09,
                        258.8,
                        259.13,
                        260.01,
                        260.81,
                        259.72,
                        257.97,
                        255.29,
                        246.47,
                        247.42,
                        248.12,
                        247.81,
                        255.17,
                        258.03,
                        256.2,
                        258.04,
                        259.24,
                        269.76,
                        269.23,
                        276.23,
                        275.65,
                        277.86,
                        274.62,
                        273.68,
                        275.5,
                        261.73,
                        255.78,
                        263.88,
                        264.35,
                    ]
                }
            ]
        }
    }


class PrevisaoResponse(BaseModel):
    """Resposta do endpoint de previsao."""

    preco_previsto: float = Field(
        ...,
        description="Preco de fechamento previsto para o proximo dia (em USD)",
    )
    simbolo: str = Field(
        ...,
        description="Simbolo da acao usada no modelo",
    )
    modelo_versao: str = Field(
        default="1.0.0",
        description="Versao do modelo que gerou a previsao",
    )


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str
    modelo_carregado: bool
    versao: str
