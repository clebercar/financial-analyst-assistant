"""Testes do gate de qualidade pos-treino."""

import json
from pathlib import Path

import pytest
import yaml

from src.models.evaluate_gate import (
    avaliar_thresholds,
    carregar_metricas,
    main,
)


def _escrever_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _escrever_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_passa_quando_todas_metricas_dentro_do_threshold():
    thresholds = {"mae_max": 10.0, "rmse_max": 15.0, "mape_max": 8.0}
    metricas = {"mae": 5.0, "rmse": 7.0, "mape": 4.0}

    violacoes = avaliar_thresholds(metricas, thresholds)

    assert violacoes == []


def test_falha_quando_metrica_max_excede():
    thresholds = {"mae_max": 10.0}
    metricas = {"mae": 12.5}

    violacoes = avaliar_thresholds(metricas, thresholds)

    assert len(violacoes) == 1
    v = violacoes[0]
    assert v.metrica == "mae"
    assert v.valor == 12.5
    assert v.threshold == 10.0
    assert v.regra == "max"


def test_falha_quando_metrica_min_abaixo():
    thresholds = {"accuracy_min": 0.70}
    metricas = {"accuracy": 0.55}

    violacoes = avaliar_thresholds(metricas, thresholds)

    assert len(violacoes) == 1
    v = violacoes[0]
    assert v.metrica == "accuracy"
    assert v.regra == "min"


def test_threshold_com_sufixo_invalido_levanta_erro():
    thresholds = {"mae_avg": 10.0}
    metricas = {"mae": 5.0}

    with pytest.raises(ValueError, match="sufixo"):
        avaliar_thresholds(metricas, thresholds)


def test_metrica_ausente_no_json_levanta_erro():
    thresholds = {"mae_max": 10.0}
    metricas = {"rmse": 7.0}  # sem 'mae'

    with pytest.raises(KeyError, match="mae"):
        avaliar_thresholds(metricas, thresholds)


def test_carregar_metricas_falha_se_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_metricas(tmp_path / "nao_existe.json")


def test_main_exit_zero_quando_passa(tmp_path, monkeypatch, capsys):
    config = {"thresholds": {"lstm": {"mae_max": 10.0}}}
    _escrever_yaml(tmp_path / "model_config.yaml", config)
    _escrever_json(tmp_path / "metrics_lstm.json", {"mae": 5.0})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_gate",
            "--model",
            "lstm",
            "--config",
            "model_config.yaml",
            "--metrics-dir",
            ".",
        ],
    )

    rc = main()

    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_main_exit_um_quando_viola(tmp_path, monkeypatch, capsys):
    config = {"thresholds": {"lstm": {"mae_max": 10.0}}}
    _escrever_yaml(tmp_path / "model_config.yaml", config)
    _escrever_json(tmp_path / "metrics_lstm.json", {"mae": 12.5})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_gate",
            "--model",
            "lstm",
            "--config",
            "model_config.yaml",
            "--metrics-dir",
            ".",
        ],
    )

    rc = main()

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "mae" in out
