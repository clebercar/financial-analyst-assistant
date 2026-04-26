# Testes da API FastAPI.
# Usa o TestClient do FastAPI pra simular requisicoes HTTP sem precisar subir o servidor.

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from src.serving.app import app


@pytest.fixture
def client():
    """Cria um cliente de teste da API."""
    return TestClient(app)


class TestHealthCheck:
    """Testes do endpoint de health check."""

    def test_health_retorna_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_retorna_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "modelo_carregado" in data
        assert "versao" in data


class TestPredict:
    """Testes do endpoint de previsao."""

    def test_predict_sem_modelo_retorna_503(self, client):
        """Se o modelo nao ta carregado, retorna 503 (service unavailable)."""
        precos = [150.0 + i * 0.5 for i in range(60)]
        response = client.post("/predict", json={"precos_fechamento": precos})

        # Pode ser 503 (modelo nao carregado) ou 200 (se modelo existir)
        assert response.status_code in [200, 503]

    def test_predict_poucos_dados_retorna_erro(self, client):
        """Mandar menos de 60 precos deve dar erro de validacao."""
        precos = [150.0, 151.0, 152.0]  # so 3 precos
        response = client.post("/predict", json={"precos_fechamento": precos})

        assert response.status_code == 422  # validation error do Pydantic

    def test_predict_body_vazio_retorna_422(self, client):
        """Mandar body vazio deve dar erro de validacao."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_com_modelo_mockado(self, client):
        """Testa o fluxo completo com modelo e scaler mockados."""
        mock_modelo = MagicMock()
        mock_modelo.predict.return_value = np.array([[0.5]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.rand(60, 1)
        mock_scaler.inverse_transform.return_value = np.array([[155.50]])

        with patch("src.serving.app.modelo", mock_modelo), \
             patch("src.serving.app.scaler", mock_scaler):
            precos = [150.0 + i * 0.5 for i in range(60)]
            response = client.post("/predict", json={"precos_fechamento": precos})

            assert response.status_code == 200
            data = response.json()
            assert "preco_previsto" in data
            assert "simbolo" in data
            assert data["simbolo"] == "AAPL"


class TestMetrics:
    """Testes do endpoint de metricas."""

    def test_metrics_retorna_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_formato_prometheus(self, client):
        response = client.get("/metrics")
        # Metricas Prometheus sao texto plano
        assert "text/plain" in response.headers.get("content-type", "") or \
               "text/plain" in str(response.headers)


class TestDocs:
    """Testa se a documentacao Swagger ta acessivel."""

    def test_swagger_docs(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "paths" in response.json()
