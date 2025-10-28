import pytest
from fastapi.testclient import TestClient

class TestEmotionAPI:
    """Тесты для Emotion Detection API"""

    def test_root(self, client: TestClient):
        """Тестирование корневого эндпоинта"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "Emotion detection API is running"

    def test_healthcheck(self, client: TestClient):
        """Тестирование эндпоинта состояния модели"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "Model is healthy!"

    def test_info(self, client: TestClient):
        """Тестирование эндпоинта получения информации о модели"""
        response = client.get("/info")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_examples(self, client: TestClient):
        """Тестирование эндпоинта получения примеров
        входных и выходных данных модели"""
        response = client.get("/examples")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_predict(self, client: TestClient, sample_texts):
        """Тестирование эндпоинта predict"""
        test_text = sample_texts["love"]
        response = client.post("/predict", json={"text": test_text})

        assert response.status_code == 200
        data = response.json()

        assert "request" in data
        assert "prediction" in data
        assert "confidence" in data

        assert isinstance(data["request"], str)
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], float)

        assert data["request"] == test_text
        assert 0 <= data["confidence"] <= 1

    def test_predict_empty(self, client: TestClient):
        """Тестирование эндпоинта predict с пустым текстом"""
        test_text = ""
        response = client.post("/predict", json={"text": test_text})
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_smiles(self, client: TestClient):
        """Тестирование эндпоинта predict со специальными символами"""
        test_cases = [
            "Hello!!! 😊",
            "What??? 😠",
            "I'm so happy :)",
            "This is sad :("
        ]

        for text in test_cases:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
            data = response.json()
            assert data["request"] == text

    def test_predict_batch(self, client: TestClient):
        """Тестирование эндпоинта predict_batch"""
        test_data = [
            "The way you care for our children makes my heart overflow with love.",
            "I can't believe I actually won the lottery, this is unbelievable!",
            "I have a bad feeling about this dark alley, let's turn back.",
            "How dare you go through my personal belongings without permission!",
            "I miss you so much, life isn't the same without you here."
        ]

        response = client.post("/predict_batch", json={"request": test_data})

        assert response.status_code == 200
        data = response.json()

        assert "prediction_4" in data

    def test_invalid_json(self, client: TestClient):
        """Тест с невалидным JSON"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422

    def test_wrong_http_method(self, client: TestClient):
        """Тест с неправильным HTTP методом"""
        response = client.get("/predict")
        assert response.status_code == 405

        response = client.put("/predict", json={"text": "test"})
        assert response.status_code == 405

        response = client.delete("/predict")
        assert response.status_code == 405

