# Testes do modulo de pre-processamento.
# Garante que normalizacao, criacao de sequencias e split estao funcionando certo.

import numpy as np

from src.models.preprocessing import (
    criar_scaler,
    criar_sequencias,
    desnormalizar_dados,
    dividir_treino_teste,
    normalizar_dados,
)


class TestNormalizacao:
    """Testes da normalizacao e desnormalizacao dos dados."""

    def test_normalizar_fica_entre_0_e_1(self):
        dados = np.array([100, 200, 300, 400, 500], dtype=float).reshape(-1, 1)
        scaler = criar_scaler()
        normalizado = normalizar_dados(dados, scaler, treinar=True)

        assert normalizado.min() >= 0.0
        assert normalizado.max() <= 1.0

    def test_desnormalizar_volta_ao_original(self):
        dados = np.array([150.5, 200.3, 180.7], dtype=float).reshape(-1, 1)
        scaler = criar_scaler()
        normalizado = normalizar_dados(dados, scaler, treinar=True)
        resultado = desnormalizar_dados(normalizado, scaler)

        np.testing.assert_array_almost_equal(resultado, dados, decimal=5)

    def test_normalizar_aceita_array_1d(self):
        dados = np.array([100, 200, 300], dtype=float)
        scaler = criar_scaler()
        normalizado = normalizar_dados(dados, scaler, treinar=True)

        assert normalizado.shape == (3, 1)


class TestSequencias:
    """Testes da criacao de sequencias (janela deslizante)."""

    def test_criar_sequencias_formato_correto(self):
        dados = np.arange(100, dtype=float).reshape(-1, 1)
        X, y = criar_sequencias(dados, tamanho_janela=10)

        # Com 100 dados e janela de 10, devemos ter 90 amostras
        assert X.shape == (90, 10, 1)
        assert y.shape == (90,)

    def test_criar_sequencias_valores_corretos(self):
        dados = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float).reshape(-1, 1)
        X, y = criar_sequencias(dados, tamanho_janela=3)

        # Primeira sequencia: [1, 2, 3] -> 4
        np.testing.assert_array_equal(X[0, :, 0], [1, 2, 3])
        assert y[0] == 4.0

        # Ultima sequencia: [4, 5, 6] -> 7
        np.testing.assert_array_equal(X[-1, :, 0], [4, 5, 6])
        assert y[-1] == 7.0

    def test_criar_sequencias_janela_60(self):
        dados = np.arange(200, dtype=float).reshape(-1, 1)
        X, y = criar_sequencias(dados, tamanho_janela=60)

        assert X.shape[1] == 60
        assert len(y) == 140  # 200 - 60


class TestSplit:
    """Testes da divisao treino/teste."""

    def test_proporcao_padrao_80_20(self):
        dados = np.arange(100, dtype=float)
        treino, teste = dividir_treino_teste(dados)

        assert len(treino) == 80
        assert len(teste) == 20

    def test_proporcao_customizada(self):
        dados = np.arange(100, dtype=float)
        treino, teste = dividir_treino_teste(dados, proporcao_treino=0.7)

        assert len(treino) == 70
        assert len(teste) == 30

    def test_dados_nao_embaralhados(self):
        """Verifica que a ordem temporal e mantida (sem shuffle)."""
        dados = np.arange(10, dtype=float)
        treino, teste = dividir_treino_teste(dados)

        np.testing.assert_array_equal(treino, np.arange(8, dtype=float))
        np.testing.assert_array_equal(teste, np.arange(8, 10, dtype=float))
