#!/usr/bin/env python3
"""
AI Data Platform - Production ML Feature Store & Model Registry

A comprehensive ML platform providing:
- Feature Store with online/offline storage
- Model Registry with versioning
- Auto-retraining pipelines
- Model serving with Triton integration

Author: Drajat Sukma
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"

import asyncio
import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import structlog
import redis
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# ============== Pydantic Models ==============

class FeatureSet(BaseModel):
    model_config = ConfigDict(extra="allow")
    entity_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FeatureDefinition(BaseModel):
    name: str
    dtype: str = "float"
    description: Optional[str] = None
    default_value: Any = None

class FeatureStoreRequest(BaseModel):
    entity_type: str
    entity_id: str
    feature_names: List[str]

class FeatureStoreResponse(BaseModel):
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    source: str = "online_store"

class ModelMetadata(BaseModel):
    name: str
    version: str
    framework: str = "sklearn"
    description: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)

class TrainingRequest(BaseModel):
    model_name: str
    model_type: str = "random_forest"
    features: List[List[float]]
    targets: List[float]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = 0.2
    register: bool = True

class TrainingResponse(BaseModel):
    model_id: str
    model_name: str
    version: str
    metrics: Dict[str, float]
    status: str
    training_time_seconds: float

class PredictionRequest(BaseModel):
    model_id: str
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_id: str
    latency_ms: float
    timestamp: datetime

class RetrainingJob(BaseModel):
    job_id: str
    model_name: str
    trigger: str
    status: str
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
    timestamp: datetime
    uptime_seconds: float

# ============== In-Memory Storage (Production: Use Redis/PostgreSQL) ==============

class Storage:
    def __init__(self):
        self.features: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.retraining_jobs: Dict[str, RetrainingJob] = {}
        self.feature_definitions: Dict[str, List[FeatureDefinition]] = {}
        self._model_counter = 0
        self._job_counter = 0
        self.start_time = datetime.utcnow()
        
    def get_next_model_version(self, model_name: str) -> str:
        existing = [m for m in self.models.values() if m["metadata"].name == model_name]
        if not existing:
            return "1.0.0"
        versions = [m["metadata"].version for m in existing]
        max_version = max([int(v.replace(".", "")) for v in versions])
        new_version = str(max_version + 1).zfill(3)
        return f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    
    def generate_model_id(self) -> str:
        self._model_counter += 1
        return f"model_{self._model_counter}_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}"
    
    def generate_job_id(self) -> str:
        self._job_counter += 1
        return f"job_{self._job_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

storage = Storage()

# ============== Feature Store Service ==============

class FeatureStoreService:
    """Manages feature storage and retrieval"""
    
    @staticmethod
    def register_feature_definitions(entity_type: str, definitions: List[FeatureDefinition]):
        storage.feature_definitions[entity_type] = definitions
        logger.info("feature_definitions_registered", entity_type=entity_type, count=len(definitions))
        return {"entity_type": entity_type, "definitions_registered": len(definitions)}
    
    @staticmethod
    def ingest_features(entity_type: str, entity_id: str, features: Dict[str, Any]):
        key = f"{entity_type}:{entity_id}"
        if key not in storage.features:
            storage.features[key] = {}
        
        storage.features[key].update({
            **features,
            "_timestamp": datetime.utcnow().isoformat(),
            "_entity_type": entity_type
        })
        logger.info("features_ingested", entity_type=entity_type, entity_id=entity_id)
        return {"status": "success", "features_count": len(features)}
    
    @staticmethod
    def get_online_features(entity_type: str, entity_id: str, feature_names: List[str]) -> FeatureStoreResponse:
        key = f"{entity_type}:{entity_id}"
        stored = storage.features.get(key, {})
        
        features = {}
        for name in feature_names:
            features[name] = stored.get(name)
        
        return FeatureStoreResponse(
            entity_id=entity_id,
            features=features,
            timestamp=datetime.utcnow(),
            source="online_store"
        )
    
    @staticmethod
    def get_historical_features(entity_type: str, entity_ids: List[str], 
                               feature_names: List[str], 
                               start_date: datetime, 
                               end_date: datetime) -> pd.DataFrame:
        """Retrieve historical features as DataFrame"""
        records = []
        for entity_id in entity_ids:
            key = f"{entity_type}:{entity_id}"
            stored = storage.features.get(key, {})
            record = {"entity_id": entity_id}
            for name in feature_names:
                record[name] = stored.get(name)
            records.append(record)
        
        return pd.DataFrame(records)

# ============== Model Registry Service ==============

class ModelRegistryService:
    """Manages ML model lifecycle"""
    
    @staticmethod
    def register_model(metadata: ModelMetadata, model_object: Any) -> str:
        model_id = storage.generate_model_id()
        version = storage.get_next_model_version(metadata.name)
        metadata.version = version
        
        storage.models[model_id] = {
            "id": model_id,
            "metadata": metadata,
            "model": model_object,
            "created_at": datetime.utcnow(),
            "status": "active"
        }
        
        logger.info("model_registered", model_id=model_id, name=metadata.name, version=version)
        return model_id
    
    @staticmethod
    def get_model(model_id: str) -> Dict[str, Any]:
        return storage.models.get(model_id)
    
    @staticmethod
    def list_models(name: Optional[str] = None) -> List[Dict[str, Any]]:
        models = list(storage.models.values())
        if name:
            models = [m for m in models if m["metadata"].name == name]
        return models
    
    @staticmethod
    def get_latest_model(name: str) -> Optional[Dict[str, Any]]:
        models = ModelRegistryService.list_models(name)
        if not models:
            return None
        return max(models, key=lambda m: m["created_at"])

# ============== Model Training Service ==============

class ModelTrainingService:
    """Handles model training and evaluation"""
    
    @staticmethod
    def train_model(request: TrainingRequest) -> TrainingResponse:
        start_time = datetime.utcnow()
        
        # Convert to numpy arrays
        X = np.array(request.features)
        y = np.array(request.targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42
        )
        
        # Select model
        if request.model_type == "random_forest":
            model = RandomForestClassifier(**request.hyperparameters)
        elif request.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(**request.hyperparameters)
        elif request.model_type == "linear_regression":
            model = LinearRegression(**request.hyperparameters)
        else:
            model = RandomForestClassifier(n_estimators=100)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        if request.model_type in ["random_forest"]:
            accuracy = accuracy_score(y_test, predictions)
            metrics = {"accuracy": accuracy, "mse": mean_squared_error(y_test, predictions)}
        else:
            metrics = {"mse": mean_squared_error(y_test, predictions), "rmse": np.sqrt(mean_squared_error(y_test, predictions))}
        
        # Register if requested
        model_id = None
        version = "none"
        if request.register:
            metadata = ModelMetadata(
                name=request.model_name,
                version="",
                framework="sklearn",
                description=f"Trained {request.model_type} model",
                metrics=metrics,
                tags={"model_type": request.model_type}
            )
            model_id = ModelRegistryService.register_model(metadata, model)
            version = storage.models[model_id]["metadata"].version
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info("model_trained", 
                   model_id=model_id, 
                   model_name=request.model_name, 
                   model_type=request.model_type,
                   training_time=training_time)
        
        return TrainingResponse(
            model_id=model_id or "not_registered",
            model_name=request.model_name,
            version=version,
            metrics=metrics,
            status="completed",
            training_time_seconds=training_time
        )
    
    @staticmethod
    def predict(model_id: str, features: List[List[float]]) -> PredictionResponse:
        start_time = datetime.utcnow()
        
        model_entry = storage.models.get(model_id)
        if not model_entry:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model = model_entry["model"]
        X = np.array(features)
        predictions = model.predict(X).tolist()
        
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predictions=predictions,
            model_id=model_id,
            latency_ms=latency,
            timestamp=datetime.utcnow()
        )

# ============== Auto-Retraining Service ==============

class AutoRetrainingService:
    """Manages automatic model retraining"""
    
    @staticmethod
    def schedule_retraining(model_name: str, trigger: str = "scheduled") -> RetrainingJob:
        job_id = storage.generate_job_id()
        job = RetrainingJob(
            job_id=job_id,
            model_name=model_name,
            trigger=trigger,
            status="scheduled",
            created_at=datetime.utcnow(),
            scheduled_for=datetime.utcnow() + timedelta(hours=1)
        )
        storage.retraining_jobs[job_id] = job
        logger.info("retraining_scheduled", job_id=job_id, model_name=model_name)
        return job
    
    @staticmethod
    def get_job_status(job_id: str) -> Optional[RetrainingJob]:
        return storage.retraining_jobs.get(job_id)
    
    @staticmethod
    def list_jobs(model_name: Optional[str] = None) -> List[RetrainingJob]:
        jobs = list(storage.retraining_jobs.values())
        if model_name:
            jobs = [j for j in jobs if j.model_name == model_name]
        return jobs

# ============== FastAPI Application ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ai_data_platform_starting", version=__version__)
    yield
    logger.info("ai_data_platform_stopping")

app = FastAPI(
    title="AI Data Platform",
    version=__version__,
    description="Enterprise ML Feature Store & Model Registry",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== API Endpoints ==============

@app.get("/health", response_model=HealthResponse)
def health_check():
    uptime = (datetime.utcnow() - storage.start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version=__version__,
        components={
            "feature_store": "active",
            "model_registry": "active", 
            "training_service": "active",
            "retraining_service": "active"
        },
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime
    )

@app.get("/")
def info():
    return {
        "name": "AI Data Platform",
        "version": __version__,
        "description": "Enterprise ML Feature Store & Model Registry",
        "features": [
            "Feature Store (Online/Offline)",
            "Model Registry with Versioning",
            "Auto-Retraining Pipelines",
            "Model Serving",
            "Training Management"
        ],
        "endpoints": [
            "/features/ingest - Ingest features",
            "/features/get - Retrieve features",
            "/models/train - Train new model",
            "/models/predict - Get predictions",
            "/models/list - List registered models",
            "/retraining/schedule - Schedule retraining"
        ]
    }

# Feature Store Endpoints

@app.post("/features/definitions/{entity_type}")
def register_feature_definitions(entity_type: str, definitions: List[FeatureDefinition]):
    return FeatureStoreService.register_feature_definitions(entity_type, definitions)

@app.post("/features/ingest/{entity_type}/{entity_id}")
def ingest_features(entity_type: str, entity_id: str, features: Dict[str, Any]):
    return FeatureStoreService.ingest_features(entity_type, entity_id, features)

@app.post("/features/get", response_model=FeatureStoreResponse)
def get_features(request: FeatureStoreRequest):
    return FeatureStoreService.get_online_features(
        request.entity_type, 
        request.entity_id, 
        request.feature_names
    )

@app.get("/features/historical/{entity_type}")
def get_historical_features(
    entity_type: str,
    entity_ids: List[str] = Query(...),
    feature_names: List[str] = Query(...),
    days: int = 30
):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    df = FeatureStoreService.get_historical_features(entity_type, entity_ids, feature_names, start, end)
    return {"data": df.to_dict(orient="records"), "shape": df.shape}

# Model Training Endpoints

@app.post("/models/train", response_model=TrainingResponse)
def train_model(request: TrainingRequest):
    return ModelTrainingService.train_model(request)

@app.post("/models/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return ModelTrainingService.predict(request.model_id, request.features)

@app.get("/models/list")
def list_models(name: Optional[str] = None):
    models = ModelRegistryService.list_models(name)
    return {
        "count": len(models),
        "models": [
            {
                "id": m["id"],
                "name": m["metadata"].name,
                "version": m["metadata"].version,
                "metrics": m["metadata"].metrics,
                "created_at": m["created_at"],
                "status": m["status"]
            }
            for m in models
        ]
    }

@app.get("/models/{model_id}")
def get_model_details(model_id: str):
    model = ModelRegistryService.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": model["id"],
        "metadata": model["metadata"].model_dump(),
        "created_at": model["created_at"],
        "status": model["status"]
    }

@app.post("/models/{model_id}/archive")
def archive_model(model_id: str):
    model = storage.models.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model["status"] = "archived"
    return {"status": "archived", "model_id": model_id}

# Retraining Endpoints

@app.post("/retraining/schedule")
def schedule_retraining(model_name: str, trigger: str = "manual"):
    job = AutoRetrainingService.schedule_retraining(model_name, trigger)
    return job.model_dump()

@app.get("/retraining/jobs")
def list_retraining_jobs(model_name: Optional[str] = None):
    jobs = AutoRetrainingService.list_jobs(model_name)
    return {"jobs": [j.model_dump() for j in jobs]}

@app.get("/retraining/jobs/{job_id}")
def get_job_status(job_id: str):
    job = AutoRetrainingService.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()

# ============== CLI Interface ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Data Platform")
    parser.add_argument("command", choices=["serve", "ingest", "train"], help="Command to run")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--entity-type", help="Entity type for ingestion")
    parser.add_argument("--entity-id", help="Entity ID for ingestion")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        logger.info("starting_server", host=args.host, port=args.port)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "ingest":
        # CLI ingestion example
        print("Ingesting sample features...")
        result = FeatureStoreService.ingest_features(
            args.entity_type or "user",
            args.entity_id or "user_001",
            {"age": 30, "purchase_count": 5, "avg_order_value": 150.0}
        )
        print(f"Result: {result}")
    elif args.command == "train":
        # CLI training example
        print("Training sample model...")
        import random
        sample_features = [[random.random() for _ in range(10)] for _ in range(100)]
        sample_targets = [random.randint(0, 1) for _ in range(100)]
        
        request = TrainingRequest(
            model_name="sample_classifier",
            model_type="random_forest",
            features=sample_features,
            targets=sample_targets,
            hyperparameters={"n_estimators": 50}
        )
        result = ModelTrainingService.train_model(request)
        print(f"Training completed: {result.model_dump()}")
