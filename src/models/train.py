"""Pipeline de treinamento com MLflow tracking padronizado.

Treina LSTM PyTorch ou classificador de sentimento sklearn dependendo do
argumento --model. Loga tudo no MLflow seguindo schema padronizado:

Tags obrigatorias em TODA run:
- model_name, model_version, model_type
- training_data_version
- owner, risk_level, fairness_checked
- git_sha, phase

Metricas (regressao):
- mae, rmse, mape (em escala desnormalizada, USD)

Artifacts:
- models/lstm_torch.pt (state_dict do modelo)
- models/scaler.joblib (MinMaxScaler ajustado)

Uso:
    python -m src.models.train --model lstm
    python -m src.models.train --model sentiment
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.data.collector import baixar_dados_acao
from src.models.lstm_torch import LSTMRegressor
from src.models.preprocessing import criar_scaler, criar_sequencias, normalizar_dados

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/model_config.yaml")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lstm_torch.pt"
SCALER_PATH = MODEL_DIR / "scaler.joblib"


def _git_sha() -> str:
    """Pega o hash curto do commit atual pra rastreabilidade da run."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:7]
        )
    except Exception:
        return "unknown"


def _set_standard_tags(model_type: str, model_name: str, training_data_version: str) -> None:
    """Aplica o schema padronizado de tags em toda run.

    Toda run tem que ter essas tags. Se alguma faltar, o run nao e considerado
    valido para deploy/monitoramento.
    """
    mlflow.set_tags(
        {
            "model_name": model_name,
            "model_version": "0.1.0",
            "model_type": model_type,
            "training_data_version": training_data_version,
            "owner": "ml-team",
            "risk_level": "high",
            "fairness_checked": "false",
            "git_sha": _git_sha(),
            "phase": "production-mvp",
        }
    )


def train_lstm(config: dict) -> str:
    """Treina o LSTM PyTorch e loga tudo no MLflow.

    Pipeline:
    1. Baixa dados via yfinance
    2. Normaliza precos com MinMaxScaler
    3. Cria sequencias com janela deslizante (sequence_length)
    4. Split treino/teste (sem shuffle - serie temporal!)
    5. Treina por N epochs
    6. Avalia em escala desnormalizada (MAE/RMSE/MAPE em USD)
    7. Salva state_dict (.pt) e scaler (.joblib)
    8. Loga params, metricas, tags e artifacts no MLflow

    Args:
        config: dict carregado do model_config.yaml (espera chave 'lstm').

    Returns:
        run_id da run criada no MLflow.
    """
    cfg = config["lstm"]

    # --- 1. Coleta ---
    df = baixar_dados_acao(cfg["ticker"], cfg["start_date"], cfg["end_date"])
    precos = df["Close"].values.reshape(-1, 1)

    # --- 2. Normalizacao ---
    scaler = criar_scaler()
    precos_norm = normalizar_dados(precos, scaler, treinar=True)

    # --- 3. Sequencias ---
    X, y = criar_sequencias(precos_norm, cfg["sequence_length"])

    # --- 4. Split temporal (sem shuffle!) ---
    split = int(len(X) * (1 - cfg["test_size"]))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    logger.info(
        "Split: treino=%d teste=%d (test_size=%.2f)", len(X_train), len(X_test), cfg["test_size"]
    )

    # Tensores PyTorch (float32). y precisa virar coluna pra match com saida do modelo.
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().reshape(-1, 1)
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().reshape(-1, 1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    # shuffle=False mantem a ordem temporal (mesmo dentro do treino).
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False)

    # --- 5. Modelo ---
    model = LSTMRegressor(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        dense_size=cfg["dense_size"],
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    mlflow.set_experiment("lstm-precos-acoes")
    with mlflow.start_run(run_name="lstm-aapl") as run:
        mlflow.log_params(cfg)
        _set_standard_tags(
            model_type="regression",
            model_name="lstm_aapl",
            training_data_version=f"{cfg['ticker']}_{cfg['start_date']}_{cfg['end_date']}",
        )

        # --- 6. Loop de treino ---
        for epoch in range(cfg["epochs"]):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg, step=epoch)
            if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
                logger.info("Epoch %d/%d - loss=%.6f", epoch + 1, cfg["epochs"], avg)

        # --- 7. Avaliacao em escala desnormalizada ---
        model.eval()
        with torch.no_grad():
            y_pred_norm = model(X_test_t).numpy()
        y_pred = scaler.inverse_transform(y_pred_norm)
        y_true = scaler.inverse_transform(y_test_t.numpy())

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # MAPE: protege contra divisao por zero (improvavel em precos, mas seguro)
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        logger.info("Test MAE=%.4f USD | RMSE=%.4f USD | MAPE=%.2f%%", mae, rmse, mape)

        # Persiste metricas em JSON para o gate de qualidade ler
        metrics_path = MODEL_DIR / "metrics_lstm.json"
        metrics_path.parent.mkdir(exist_ok=True)
        metrics_path.write_text(
            json.dumps({"mae": mae, "rmse": rmse, "mape": mape}), encoding="utf-8"
        )
        mlflow.log_artifact(str(metrics_path))

        # --- 8. Salvar artifacts ---
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SCALER_PATH))
        logger.info("Modelo salvo em %s | Scaler em %s", MODEL_PATH, SCALER_PATH)

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de treino dos modelos baseline")
    parser.add_argument(
        "--model",
        choices=["lstm", "sentiment"],
        required=True,
        help="Qual modelo treinar (lstm = preco de acao, sentiment = polaridade de noticia).",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Override do ticker no config (so faz sentido com --model lstm).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Override de start_date no formato YYYY-MM-DD (so --model lstm).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Override de end_date no formato YYYY-MM-DD (so --model lstm).",
    )
    args = parser.parse_args()

    if args.model == "sentiment" and (args.ticker or args.start_date or args.end_date):
        parser.error("--ticker/--start-date/--end-date so se aplicam a --model lstm")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Aplica overrides na secao do LSTM (mantem retrocompat: sem flag = usa YAML)
    if args.model == "lstm":
        if args.ticker:
            config["lstm"]["ticker"] = args.ticker
        if args.start_date:
            config["lstm"]["start_date"] = args.start_date
        if args.end_date:
            config["lstm"]["end_date"] = args.end_date

    if args.model == "lstm":
        run_id = train_lstm(config)
        logger.info("LSTM treinado com sucesso. run_id=%s", run_id)
    elif args.model == "sentiment":
        from src.models.sentiment_classifier import train_sentiment

        run_id = train_sentiment(config)
        logger.info("Sentiment treinado com sucesso. run_id=%s", run_id)


if __name__ == "__main__":
    main()
