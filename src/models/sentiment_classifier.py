"""Classificador de sentimento de texto financeiro (TF-IDF + LogReg).

Pipeline simples mas competitivo pra texto curto/medio:
  TfidfVectorizer (n-grams 1-2) -> LogisticRegression (class_weight=balanced)

Por que TF-IDF + LogReg em vez de transformer?
- Dataset pequeno (~3.500 frases) - transformer iria overfitar facil
- Treino em segundos vs. minutos
- Interpretavel: da pra inspecionar coeficientes por feature
- Bom o suficiente: F1 macro ~70-75% no FinancialPhraseBank
- Sem GPU necessaria pra treino nem pra inferencia

Tags MLflow padronizadas:
- model_type=classification
- risk_level=medium (errar sentimento de noticia nao tem o mesmo impacto
  de errar previsao de preco usada em ordem de compra)
"""

import logging
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.financial_phrasebank import load_phrasebank
from src.features.feature_engineering import validate_sentiment_input

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_classifier.joblib"


def _git_sha() -> str:
    """Hash curto do commit atual pra rastreabilidade da run no MLflow."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:7]
        )
    except Exception:
        return "unknown"


def predict_sentiment(text: str, clf: Pipeline) -> dict:
    """Prediz sentimento de um texto usando o pipeline injetado.

    Args:
        text: frase a ser classificada (texto livre, sem pre-processamento).
        clf: pipeline ja treinado (TfidfVectorizer + LogisticRegression).
            Pode ser um objeto sklearn real ou um mock para testes.

    Returns:
        Dict com:
            - 'sentimento' (str): 'positive', 'neutral' ou 'negative'
            - 'confianca' (float): probabilidade da classe predita, em [0, 1]
    """
    label = clf.predict([text])[0]
    proba = clf.predict_proba([text])[0]
    confidence = float(np.max(proba))
    return {"sentimento": label, "confianca": confidence}


def train_sentiment(config: dict) -> str:
    """Treina o classificador de sentimento e loga tudo no MLflow.

    Pipeline:
    1. Carrega FinancialPhraseBank (com cache do HF datasets)
    2. Valida com schema Pandera
    3. Stratified split treino/teste
    4. Fita TF-IDF + LogReg (class_weight=balanced pra labels desbalanceados)
    5. Avalia com F1/precision/recall macro
    6. Loga params, metricas, tags, artifacts e classification_report

    Args:
        config: dict do model_config.yaml (precisa ter chave 'sentiment').

    Returns:
        run_id da run criada no MLflow.
    """
    cfg = config["sentiment"]

    # --- 1. Carregar dataset ---
    df = load_phrasebank(cfg["config"])
    # --- 2. Validacao de schema (falha cedo se vier algo estranho) ---
    df = validate_sentiment_input(df)

    # --- 3. Split estratificado (mantem proporcao das 3 classes em treino e teste) ---
    X_train, X_test, y_train, y_test = train_test_split(
        df["sentence"].values,
        df["label"].values,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df["label"].values,
    )
    logger.info("Split: treino=%d teste=%d", len(X_train), len(X_test))

    # --- 4. Pipeline TF-IDF + LogReg ---
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(cfg["vectorizer"]["ngram_range"]),
                    max_features=cfg["vectorizer"]["max_features"],
                    min_df=cfg["vectorizer"]["min_df"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=cfg["random_state"],
                ),
            ),
        ]
    )

    mlflow.set_experiment("sentimento-financeiro")
    with mlflow.start_run(run_name="tfidf-logreg") as run:
        # --- Params ---
        mlflow.log_params(
            {
                "dataset": cfg["dataset"],
                "subset": cfg["config"],
                "test_size": cfg["test_size"],
                "random_state": cfg["random_state"],
                "ngram_range": cfg["vectorizer"]["ngram_range"],
                "max_features": cfg["vectorizer"]["max_features"],
                "min_df": cfg["vectorizer"]["min_df"],
            }
        )
        # --- Tags padronizadas (mesmo schema do LSTM, ajustando para classificacao) ---
        mlflow.set_tags(
            {
                "model_name": "sentiment_phrasebank",
                "model_version": "0.1.0",
                "model_type": "classification",
                "training_data_version": cfg["config"],
                "owner": "ml-team",
                "risk_level": "medium",
                "fairness_checked": "false",
                "git_sha": _git_sha(),
                "phase": "production-mvp",
            }
        )

        # --- 5. Treino + avaliacao ---
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "precision_macro": float(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        }
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")
        logger.info(
            "Sentimento F1=%.4f Precision=%.4f Recall=%.4f",
            metrics["f1_macro"],
            metrics["precision_macro"],
            metrics["recall_macro"],
        )

        # --- 6. Salvar artifacts ---
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        logger.info("Modelo salvo em %s", MODEL_PATH)

        return run.info.run_id
