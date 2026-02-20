# Modulo de treinamento e avaliacao do modelo.
# Aqui junta tudo: coleta dados, preprocessa, treina o LSTM e calcula as metricas.

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica (pra funcionar em servidor/Docker)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from src.data.collector import baixar_dados_acao, extrair_preco_fechamento
from src.model.preprocessing import (
    criar_scaler,
    normalizar_dados,
    desnormalizar_dados,
    criar_sequencias,
    dividir_treino_teste,
)
from src.model.lstm import criar_modelo_lstm


# Caminhos padrao pra salvar os artefatos do modelo
DIRETORIO_MODELOS = Path("models")
CAMINHO_MODELO = DIRETORIO_MODELOS / "lstm_model.keras"
CAMINHO_SCALER = DIRETORIO_MODELOS / "scaler.joblib"
CAMINHO_GRAFICO = DIRETORIO_MODELOS / "previsao_vs_real.png"


def calcular_mape(y_real: np.ndarray, y_previsto: np.ndarray) -> float:
    """
    Calcula o MAPE (Mean Absolute Percentage Error).
    Indica o erro medio em percentual - facil de interpretar.
    Ex: MAPE de 5% significa que, em media, a previsao erra 5% do valor real.
    """
    # Evita divisao por zero filtrando valores muito proximos de 0
    mascara = y_real != 0
    return float(np.mean(np.abs((y_real[mascara] - y_previsto[mascara]) / y_real[mascara])) * 100)


def treinar_modelo(
    simbolo: str = "AAPL",
    data_inicio: str = "2018-01-01",
    data_fim: str = "2024-12-31",
    tamanho_janela: int = 60,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Pipeline completo de treinamento:
    1. Baixa os dados
    2. Normaliza
    3. Cria sequencias
    4. Treina o LSTM
    5. Avalia com metricas
    6. Salva modelo e scaler

    Retorna dict com as metricas de avaliacao.
    """
    # --- 1. Coleta de dados ---
    df = baixar_dados_acao(simbolo, data_inicio, data_fim)
    precos = extrair_preco_fechamento(df)
    valores = precos.values.reshape(-1, 1)

    # --- 2. Normalizacao ---
    scaler = criar_scaler()
    dados_normalizados = normalizar_dados(valores, scaler, treinar=True)

    # --- 3. Split treino/teste ---
    dados_treino, dados_teste = dividir_treino_teste(dados_normalizados)

    # --- 4. Criar sequencias ---
    X_treino, y_treino = criar_sequencias(dados_treino, tamanho_janela)
    X_teste, y_teste = criar_sequencias(dados_teste, tamanho_janela)

    print(f"Shape do X_treino: {X_treino.shape} (amostras, janela, features)")
    print(f"Shape do X_teste: {X_teste.shape}")

    # --- 5. Criar e treinar o modelo ---
    modelo = criar_modelo_lstm(tamanho_janela)
    modelo.summary()

    # EarlyStopping: para o treinamento se o loss do validation nao melhorar
    # por 5 epochs seguidas. Evita treinar a toa e overfitting.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,  # volta pros pesos da melhor epoch
    )

    historico = modelo.fit(
        X_treino,
        y_treino,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # 10% do treino vira validacao
        callbacks=[early_stop],
        verbose=1,
    )

    # --- 6. Avaliacao no conjunto de teste ---
    previsoes_normalizadas = modelo.predict(X_teste)

    # Volta pra escala original pra calcular metricas em dolares
    previsoes = desnormalizar_dados(previsoes_normalizadas, scaler)
    reais = desnormalizar_dados(y_teste.reshape(-1, 1), scaler)

    mae = mean_absolute_error(reais, previsoes)
    rmse = float(np.sqrt(mean_squared_error(reais, previsoes)))
    mape = calcular_mape(reais.flatten(), previsoes.flatten())

    metricas = {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}

    print("\n--- Metricas de Avaliacao ---")
    print(f"MAE  (Erro Absoluto Medio):       ${metricas['mae']:.2f}")
    print(f"RMSE (Raiz do Erro Quadratico):   ${metricas['rmse']:.2f}")
    print(f"MAPE (Erro Percentual Medio):     {metricas['mape']:.2f}%")

    # --- 7. Salvar artefatos ---
    DIRETORIO_MODELOS.mkdir(exist_ok=True)

    modelo.save(CAMINHO_MODELO)
    print(f"\nModelo salvo em: {CAMINHO_MODELO}")

    joblib.dump(scaler, CAMINHO_SCALER)
    print(f"Scaler salvo em: {CAMINHO_SCALER}")

    # --- 8. Gerar grafico de comparacao ---
    _gerar_grafico(reais, previsoes, historico)

    return metricas


def _gerar_grafico(reais: np.ndarray, previsoes: np.ndarray, historico) -> None:
    """Gera graficos de previsao vs real e de evolucao do loss durante o treino."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Grafico 1: Preco real vs previsto
    axes[0].plot(reais, label="Preco Real", color="blue", linewidth=1.5)
    axes[0].plot(previsoes, label="Previsao LSTM", color="red", linewidth=1.5, alpha=0.8)
    axes[0].set_title("Previsao vs Preco Real (Conjunto de Teste)")
    axes[0].set_xlabel("Dias")
    axes[0].set_ylabel("Preco (USD)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Grafico 2: Loss durante o treinamento
    axes[1].plot(historico.history["loss"], label="Loss Treino", color="blue")
    axes[1].plot(historico.history["val_loss"], label="Loss Validacao", color="red")
    axes[1].set_title("Evolucao do Loss durante o Treinamento")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CAMINHO_GRAFICO, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafico salvo em: {CAMINHO_GRAFICO}")


# Permite rodar direto com: python -m src.model.trainer
if __name__ == "__main__":
    treinar_modelo()
