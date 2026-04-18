#!/usr/bin/env python3
"""Quickstart example for AI Data Platform"""

import requests
import random
import time

BASE_URL = "http://localhost:8000"


def main():
    print("=== AI Data Platform Quickstart ===\n")
    
    # 1. Check health
    print("1. Checking platform health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.json()['status']}")
    print(f"   Components: {response.json()['components']}\n")
    
    # 2. Register feature definitions
    print("2. Registering feature definitions...")
    definitions = [
        {"name": "age", "dtype": "int", "description": "Customer age"},
        {"name": "income", "dtype": "float", "description": "Annual income"},
        {"name": "purchase_history", "dtype": "int", "description": "Number of past purchases"},
        {"name": "avg_order_value", "dtype": "float", "description": "Average order value"}
    ]
    response = requests.post(f"{BASE_URL}/features/definitions/customer", json=definitions)
    print(f"   Registered {response.json()['definitions_registered']} features\n")
    
    # 3. Ingest customer features
    print("3. Ingesting customer features...")
    for i in range(10):
        customer_id = f"customer_{i:03d}"
        features = {
            "age": random.randint(18, 70),
            "income": random.uniform(30000, 150000),
            "purchase_history": random.randint(0, 50),
            "avg_order_value": random.uniform(20, 500)
        }
        requests.post(f"{BASE_URL}/features/ingest/customer/{customer_id}", json=features)
    print("   Ingested features for 10 customers\n")
    
    # 4. Retrieve features
    print("4. Retrieving features for customer_001...")
    request = {
        "entity_type": "customer",
        "entity_id": "customer_001",
        "feature_names": ["age", "income", "purchase_history"]
    }
    response = requests.post(f"{BASE_URL}/features/get", json=request)
    print(f"   Features: {response.json()['features']}\n")
    
    # 5. Train a model
    print("5. Training customer churn prediction model...")
    # Generate synthetic training data
    features = [[random.random() * 100 for _ in range(4)] for _ in range(200)]
    targets = [random.randint(0, 1) for _ in range(200)]
    
    train_request = {
        "model_name": "churn_predictor",
        "model_type": "random_forest",
        "features": features,
        "targets": targets,
        "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        "test_size": 0.2,
        "register": True
    }
    response = requests.post(f"{BASE_URL}/models/train", json=train_request)
    result = response.json()
    print(f"   Model ID: {result['model_id']}")
    print(f"   Version: {result['version']}")
    print(f"   Metrics: {result['metrics']}")
    print(f"   Training time: {result['training_time_seconds']:.2f}s\n")
    
    model_id = result['model_id']
    
    # 6. Make predictions
    print("6. Making predictions...")
    predict_request = {
        "model_id": model_id,
        "features": [
            [45.0, 80000.0, 10.0, 150.0],  # High-value customer
            [25.0, 35000.0, 2.0, 25.0]      # Low-value customer
        ]
    }
    response = requests.post(f"{BASE_URL}/models/predict", json=predict_request)
    predictions = response.json()['predictions']
    print(f"   Customer 1 churn probability: {predictions[0]:.2f}")
    print(f"   Customer 2 churn probability: {predictions[1]:.2f}\n")
    
    # 7. List registered models
    print("7. Listing registered models...")
    response = requests.get(f"{BASE_URL}/models/list")
    models = response.json()['models']
    print(f"   Total models: {len(models)}")
    for m in models:
        print(f"   - {m['name']} v{m['version']} ({m['status']})")
    
    # 8. Schedule retraining
    print("\n8. Scheduling automatic retraining...")
    response = requests.post(f"{BASE_URL}/retraining/schedule?model_name=churn_predictor&trigger=scheduled")
    job = response.json()
    print(f"   Job ID: {job['job_id']}")
    print(f"   Status: {job['status']}")
    print(f"   Scheduled for: {job['scheduled_for']}\n")
    
    print("=== Quickstart completed! ===")
    print("\nNext steps:")
    print("- Explore the API docs at http://localhost:8000/docs")
    print("- View metrics and logs")
    print("- Schedule regular retraining jobs")


if __name__ == "__main__":
    main()
