# Modulo de pre-processamento dos dados.
# Aqui a gente transforma os dados brutos em algo que a rede neural consegue digerir:
# - Normaliza os valores pra ficar entre 0 e 1 (redes neurais funcionam melhor assim)
# - Cria as sequencias de entrada (janela deslizante)
# - Divide em treino e teste


import numpy as np
from sklearn.preprocessing import MinMaxScaler


def criar_scaler() -> MinMaxScaler:
    """Cria o normalizador que vai escalar os dados entre 0 e 1."""
    return MinMaxScaler(feature_range=(0, 1))


def normalizar_dados(dados: np.ndarray, scaler: MinMaxScaler, treinar: bool = True) -> np.ndarray:
    """
    Normaliza os dados pra ficar no intervalo [0, 1].

    Por que fazer isso? Porque valores de acoes variam de dezenas a milhares.
    Se a gente nao normalizar, a rede neural tem dificuldade pra convergir
    (os gradientes ficam muito grandes ou muito pequenos).

    Parametros:
        dados: array com os precos (shape: (n, 1))
        scaler: o normalizador MinMaxScaler
        treinar: se True, ajusta o scaler nos dados (fit_transform).
                 se False, so aplica a transformacao ja aprendida (transform).
                 Importante: no teste a gente usa False pra nao ter data leakage.
    """
    if dados.ndim == 1:
        dados = dados.reshape(-1, 1)

    if treinar:
        return scaler.fit_transform(dados)
    else:
        return scaler.transform(dados)


def desnormalizar_dados(dados: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Reverte a normalizacao, trazendo os valores de volta pra escala original (em reais/dolares).
    Necessario pra mostrar os resultados de um jeito que faca sentido.
    """
    if dados.ndim == 1:
        dados = dados.reshape(-1, 1)

    return scaler.inverse_transform(dados)


def criar_sequencias(dados: np.ndarray, tamanho_janela: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria as sequencias de entrada e saida usando janela deslizante.

    A ideia e simples: pra prever o preco do dia 61, a gente usa os precos
    dos 60 dias anteriores como entrada. Ai desliza a janela um dia pra frente
    e repete.

    Exemplo com janela de 3:
        dados = [10, 20, 30, 40, 50]
        X = [[10, 20, 30], [20, 30, 40]]  -> entradas
        y = [40, 50]                        -> saidas (o que queremos prever)

    Parametros:
        dados: array normalizado com os precos
        tamanho_janela: quantos dias anteriores usar como entrada (default: 60)

    Retorna:
        X: array de shape (n_amostras, tamanho_janela, 1) -> entrada do LSTM
        y: array de shape (n_amostras,) -> valor a ser previsto
    """
    X, y = [], []

    for i in range(tamanho_janela, len(dados)):
        X.append(dados[i - tamanho_janela:i, 0])
        y.append(dados[i, 0])

    X = np.array(X)
    y = np.array(y)

    # O LSTM espera entrada 3D: (amostras, timesteps, features)
    # No nosso caso: (n_amostras, 60, 1) - 1 feature porque so usamos o preco de fechamento
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


def dividir_treino_teste(
    dados: np.ndarray,
    proporcao_treino: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide os dados em treino e teste.

    Importante: a gente NAO embaralha os dados! Em series temporais, a ordem importa.
    O treino usa os dados mais antigos e o teste usa os mais recentes,
    simulando a situacao real onde voce treina com o passado e testa no futuro.

    Parametros:
        dados: array com todos os precos normalizados
        proporcao_treino: fracao dos dados pra treino (default: 80%)

    Retorna:
        dados_treino, dados_teste
    """
    tamanho_treino = int(len(dados) * proporcao_treino)
    dados_treino = dados[:tamanho_treino]
    dados_teste = dados[tamanho_treino:]

    print(f"Treino: {len(dados_treino)} amostras | Teste: {len(dados_teste)} amostras")

    return dados_treino, dados_teste
