#!/usr/bin/env python3
"""Tests for AI Data Platform"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from app.main import app, FeatureStoreService, ModelTrainingService, ModelRegistryService
from app.main import TrainingRequest, FeatureDefinition

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "AI Data Platform" in data["name"]


class TestFeatureStore:
    def test_register_feature_definitions(self):
        definitions = [
            {"name": "age", "dtype": "int", "description": "User age"},
            {"name": "income", "dtype": "float", "description": "Annual income"}
        ]
        response = client.post("/features/definitions/user", json=definitions)
        assert response.status_code == 200
        assert response.json()["definitions_registered"] == 2

    def test_ingest_and_get_features(self):
        # Ingest features
        features = {"age": 30, "income": 75000.0, "city": "NYC"}
        response = client.post("/features/ingest/user/user_001", json=features)
        assert response.status_code == 200
        
        # Get features
        request = {
            "entity_type": "user",
            "entity_id": "user_001",
            "feature_names": ["age", "income"]
        }
        response = client.post("/features/get", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["entity_id"] == "user_001"
        assert data["features"]["age"] == 30


class TestModelTraining:
    def test_train_model(self):
        import random
        request = {
            "model_name": "test_classifier",
            "model_type": "random_forest",
            "features": [[random.random() for _ in range(5)] for _ in range(50)],
            "targets": [random.randint(0, 1) for _ in range(50)],
            "hyperparameters": {"n_estimators": 10}
        }
        response = client.post("/models/train", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "model_id" in data
        assert "metrics" in data

    def test_list_models(self):
        response = client.get("/models/list")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_predict(self):
        # First train a model
        import random
        train_request = {
            "model_name": "predict_test",
            "model_type": "linear_regression",
            "features": [[i, i*2] for i in range(20)],
            "targets": [i * 3 for i in range(20)],
            "register": True
        }
        train_response = client.post("/models/train", json=train_request)
        model_id = train_response.json()["model_id"]
        
        # Then predict
        predict_request = {
            "model_id": model_id,
            "features": [[1.0, 2.0], [3.0, 6.0]]
        }
        response = client.post("/models/predict", json=predict_request)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2


class TestRetraining:
    def test_schedule_retraining(self):
        response = client.post("/retraining/schedule?model_name=test_model&trigger=test")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert data["status"] == "scheduled"

    def test_list_jobs(self):
        response = client.get("/retraining/jobs")
        assert response.status_code == 200
        assert "jobs" in response.json()
