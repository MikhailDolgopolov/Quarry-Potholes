import datetime
import hashlib
import os
import pickle
import json
import re
from pprint import pprint
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

MODEL_REGISTRY = "models/model_registry.json"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_registry():
    """Ensure the model registry file exists with proper structure"""
    registry_path = Path(MODEL_REGISTRY)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        with open(registry_path, 'w') as f:
            json.dump({}, f)
        logger.info(f"Initialized model registry at {registry_path}")


def sanitize_filename(filename: str) -> str:
    """Sanitize string to be filesystem-safe"""
    # Replace invalid characters with underscores
    return re.sub(r'[^a-zA-Z0-9-_.]', '_', filename)


def get_model_filename(model_type: str, params: Dict, features: List[str]) -> Path:
    """Generate filesystem-safe model filename with metadata tracking"""
    # Sanitize model type
    safe_model_type = sanitize_filename(model_type)

    # Serialize and sanitize parameters
    param_parts = []
    for key in sorted(params.keys()):  # Sort for consistency
        value = params[key]
        # Handle different parameter types
        if isinstance(value, (list, tuple)):
            str_value = ','.join(map(str, value))
        elif isinstance(value, dict):
            str_value = '-'.join(f"{k}:{v}" for k, v in value.items())
        else:
            str_value = str(value)
        param_parts.append(f"{key}={sanitize_filename(str_value)}")

    param_str = '_'.join(param_parts)

    # Create feature hash (sorted for consistency)
    feature_str = ''.join(sorted(features))
    feature_hash = hashlib.md5(feature_str.encode()).hexdigest()[:8]

    # Build filename components
    filename = f"{safe_model_type}_{param_str}_{feature_hash}.pkl"
    safe_filename = sanitize_filename(filename)

    # Ensure valid path structure
    return Path("models") / safe_model_type / safe_filename

def register_model(model_path: Path, metrics: Dict, features: List[str]):
    """Store model metadata in registry"""
    metadata = {
        'path': str(model_path),
        'features': features,  # Stored in training order
        'metrics': metrics,
        'timestamp': datetime.datetime.now().isoformat()
    }
    registry_path = Path(MODEL_REGISTRY)
    if not registry_path.exists():
        initialize_registry()
    with open(registry_path) as f:
        registry = json.load(f)
    registry[str(model_path)] = metadata
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Registered model: {model_path}")

def discover_models(
        model_type: Optional[str] = None,
        params: Optional[Dict] = None,
        min_accuracy: Optional[float] = None,
        latest: bool = False
) -> List[Dict]:
    """Find models matching criteria"""
    registry_path = Path(MODEL_REGISTRY)
    if not registry_path.exists():
        logger.warning("Model registry not found")
        return []
    with open(registry_path) as f:
        registry = json.load(f)
    candidates = []
    for meta in registry.values():
        if model_type and model_type != Path(meta['path']).parent.name:
            continue
        if params and not all(meta.get('params', {}).get(k) == v for k, v in params.items()):
            continue
        if min_accuracy and meta['metrics'].get('accuracy', 0) < min_accuracy:
            continue
        candidates.append(meta)
    if latest:
        candidates.sort(key=lambda x: x['timestamp'], reverse=True)
    else:
        candidates.sort(key=lambda x: x['metrics'].get('accuracy', 0), reverse=True)
    return candidates

def model_can_predict(model_path: Path, X: pd.DataFrame) -> tuple[bool, List[str]]:
    """Check feature compatibility and return ordered features"""
    registry_path = Path(MODEL_REGISTRY)
    if not registry_path.exists():
        raise ValueError("Model registry not found")
    with open(registry_path) as f:
        registry = json.load(f)
    if str(model_path) not in registry:
        raise ValueError(f"Model not registered: {model_path}")
    required_features = registry[str(model_path)]['features']  # Use training order
    available_features = set(X.columns)
    if not set(required_features).issubset(available_features):
        return False, required_features
    return True, required_features

def predict_with_my_model(
        X: pd.DataFrame,
        model_type: Optional[str] = None,
        params: Optional[Dict] = None,
        strategy: str = "best"
) -> np.ndarray:
    """Robust prediction with feature order consistency"""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    candidates = discover_models(model_type=model_type, params=params, latest=(strategy == "latest"))
    if not candidates:
        raise ValueError("No models found matching the criteria")
    if strategy == "latest":
        best_model_info = max(candidates, key=lambda x: x['timestamp'])
    else:
        best_model_info = max(candidates, key=lambda x: x['metrics'].get('accuracy', 0))
    model_path = Path(best_model_info['path'])
    compatible, features = model_can_predict(model_path, X)
    if not compatible:
        missing = set(features) - set(X.columns)
        raise ValueError(f"Input data missing required features: {missing}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        X_subset = X[features]  # Preserve training feature order
        return model.predict(X_subset)
    except Exception as e:
        logger.error(f"Prediction failed for {model_path}: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

def predict_with_top_models(
        X: pd.DataFrame,
        n_models: int = 4,
        model_type: Optional[str] = None,
        params: Optional[Dict] = None,
        min_accuracy: Optional[float] = None,
        return_probas: bool = False,
        strict: bool = False
) -> np.ndarray:
    """Robust multi-model prediction with consistent feature handling"""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    if n_models < 1:
        raise ValueError("n_models must be at least 1")
    candidates = discover_models(model_type=model_type, params=params, min_accuracy=min_accuracy)
    if not candidates:
        raise ValueError("No models found matching the criteria")
    if strict and len(candidates) < n_models:
        raise ValueError(f"Found only {len(candidates)} models (requested {n_models})")
    predictions = []
    for model_info in candidates[:n_models]:
        model_path = Path(model_info['path'])
        compatible, features = model_can_predict(model_path, X)
        if not compatible:
            logger.warning(f"Skipping {model_path}: incompatible features")
            continue
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            X_subset = X[features]  # Preserve training feature order
            if return_probas:
                if not hasattr(model, "predict_proba"):
                    logger.warning(f"Model {model_path} does not support predict_proba")
                    continue
                preds = model.predict_proba(X_subset)[:, 1]
            else:
                preds = model.predict(X_subset)
            predictions.append(preds)
        except Exception as e:
            logger.error(f"Prediction failed for {model_path}: {str(e)}")
            continue
    if not predictions:
        raise ValueError("No models were able to make predictions")
    return np.vstack(predictions)


def actualize_registry():
    """Remove registry entries for models that no longer exist on disk."""
    registry_path = Path(MODEL_REGISTRY)

    # Check if the registry file exists
    if not registry_path.exists():
        logger.info("Model registry not found, nothing to actualize")
        return

    # Load the registry, handling potential JSON errors
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except json.JSONDecodeError:
        logger.error("Model registry is corrupted, cannot actualize")
        return

    # Count original entries
    original_count = len(registry)

    # Filter out entries where the model file doesn’t exist
    updated_registry = {
        model_path: metadata
        for model_path, metadata in registry.items()
        if Path(model_path).is_file()
    }

    # Calculate and log the number of removed entries
    removed_count = original_count - len(updated_registry)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale entries from model registry")
    else:
        logger.info("No stale entries found in model registry")

    # Save the updated registry
    with open(registry_path, 'w') as f:
        json.dump(updated_registry, f, indent=2)

    logger.info("Model registry actualized")