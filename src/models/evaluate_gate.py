"""Gate de qualidade pos-treino. Compara metricas com thresholds do config.

Uso:
    python -m src.models.evaluate_gate --model lstm
    python -m src.models.evaluate_gate --model sentiment

Convencao de thresholds (chave em configs/model_config.yaml):
- chave terminada em '_max': metrica precisa ser <= threshold
- chave terminada em '_min': metrica precisa ser >= threshold

Exit codes:
- 0: todas as metricas dentro dos thresholds
- 1: pelo menos uma violacao (relatorio impresso em stdout)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class GateViolation:
    """Uma violacao especifica de threshold."""

    metrica: str
    valor: float
    threshold: float
    regra: str  # 'max' ou 'min'

    def __str__(self) -> str:
        op = "<=" if self.regra == "max" else ">="
        return f"{self.metrica}={self.valor:.4f} " f"(deveria ser {op} {self.threshold:.4f})"


def carregar_metricas(path: Path) -> dict[str, float]:
    """Carrega metrics_<model>.json. Levanta FileNotFoundError se ausente."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de metricas nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def carregar_thresholds(path: Path, modelo: str) -> dict[str, float]:
    """Le thresholds.<modelo> do YAML de config."""
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    thresholds = cfg.get("thresholds", {}).get(modelo)
    if not thresholds:
        raise KeyError(f"Sem thresholds.{modelo} em {path}")
    return thresholds


def avaliar_thresholds(
    metricas: dict[str, float], thresholds: dict[str, float]
) -> list[GateViolation]:
    """Compara cada threshold com a metrica correspondente.

    Para cada chave em thresholds:
    - extrai nome da metrica e regra a partir do sufixo (_max ou _min)
    - busca a metrica em metricas (KeyError se ausente)
    - aplica regra; se violar, anexa GateViolation

    Returns:
        Lista de violacoes (vazia se tudo passou).

    Raises:
        ValueError: se algum threshold nao tiver sufixo _max/_min.
        KeyError: se alguma metrica esperada nao estiver no JSON.
    """
    violacoes: list[GateViolation] = []
    for chave, limite in thresholds.items():
        if chave.endswith("_max"):
            metrica = chave[:-4]
            regra = "max"
        elif chave.endswith("_min"):
            metrica = chave[:-4]
            regra = "min"
        else:
            raise ValueError(f"Threshold '{chave}' precisa de sufixo '_max' ou '_min'")

        if metrica not in metricas:
            raise KeyError(f"Metrica '{metrica}' ausente no JSON de metricas")

        valor = float(metricas[metrica])
        violou = (regra == "max" and valor > limite) or (regra == "min" and valor < limite)
        if violou:
            violacoes.append(
                GateViolation(metrica=metrica, valor=valor, threshold=limite, regra=regra)
            )
    return violacoes


def main() -> int:
    """Ponto de entrada do gate. Retorna 0 (PASS) ou 1 (FAIL)."""
    parser = argparse.ArgumentParser(description="Gate de qualidade pos-treino")
    parser.add_argument("--model", required=True, choices=["lstm", "sentiment"])
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Caminho do YAML com a secao 'thresholds'",
    )
    parser.add_argument(
        "--metrics-dir",
        default="models",
        help="Diretorio onde fica metrics_<model>.json",
    )
    args = parser.parse_args()

    thresholds = carregar_thresholds(Path(args.config), args.model)
    metricas_path = Path(args.metrics_dir) / f"metrics_{args.model}.json"
    metricas = carregar_metricas(metricas_path)

    violacoes = avaliar_thresholds(metricas, thresholds)

    if not violacoes:
        print(f"[gate:{args.model}] PASS - todas as metricas dentro dos thresholds")
        return 0

    print(f"[gate:{args.model}] FAIL - {len(violacoes)} violacao(oes):")
    for v in violacoes:
        print(f"  - {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
