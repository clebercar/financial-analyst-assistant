# Definicao da arquitetura do modelo LSTM.
#
# LSTM (Long Short-Term Memory) e um tipo de rede neural recorrente (RNN)
# que consegue aprender padroes de longo prazo em sequencias de dados.
#
# A sacada do LSTM comparado com RNNs comuns e que ele tem um mecanismo de
# "portas" que controla o que lembrar e o que esquecer:
#   - Forget Gate: decide o que jogar fora da memoria
#   - Input Gate: decide o que guardar de novo
#   - Output Gate: decide o que mandar como saida
#
# Isso resolve o problema de "vanishing gradient" que faz RNNs simples
# esquecerem informacoes antigas durante o treinamento.

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential


def criar_modelo_lstm(tamanho_janela: int = 60, n_features: int = 1) -> Sequential:
    """
    Monta a arquitetura do modelo LSTM.

    A estrutura e a seguinte:
    1. Primeira camada LSTM com 50 neuronios - captura padroes na sequencia
       (return_sequences=True porque a proxima camada tambem e LSTM e precisa
       receber a sequencia completa, nao so o ultimo valor)
    2. Dropout de 20% - desliga neuronios aleatorios pra evitar overfitting
       (o modelo "decorar" os dados de treino ao inves de aprender padroes gerais)
    3. Segunda camada LSTM com 50 neuronios - refina os padroes capturados
       (return_sequences=False porque a proxima camada e Dense e espera um vetor)
    4. Dropout de 20% - mesma ideia de regularizacao
    5. Camada Dense com 25 neuronios - combina as features extraidas pelo LSTM
    6. Camada Dense com 1 neuronio - saida final (o preco previsto)

    Parametros:
        tamanho_janela: quantos dias anteriores o modelo recebe como entrada
        n_features: quantas variaveis por dia (1 = so preco de fechamento)

    Retorna:
        Modelo Keras compilado e pronto pra treinar
    """
    modelo = Sequential([
        # Input explicito - define o formato da entrada
        Input(shape=(tamanho_janela, n_features)),

        # Primeira camada LSTM
        LSTM(50, return_sequences=True),
        Dropout(0.2),

        # Segunda camada LSTM
        LSTM(50, return_sequences=False),
        Dropout(0.2),

        # Camadas densas pra gerar a previsao final
        Dense(25, activation="relu"),
        Dense(1),  # saida: 1 valor (preco previsto)
    ])

    # Compilacao do modelo:
    # - optimizer='adam': ajusta a taxa de aprendizado automaticamente durante o treino
    # - loss='mean_squared_error': funcao de perda pra regressao (queremos minimizar o erro quadratico)
    modelo.compile(optimizer="adam", loss="mean_squared_error")

    return modelo


def resumo_modelo(modelo: Sequential) -> None:
    """Printa um resumo da arquitetura do modelo (camadas, parametros, etc.)."""
    modelo.summary()
