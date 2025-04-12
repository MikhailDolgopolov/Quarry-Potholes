# batch_training.py
import json
import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklvq import GLVQ
from exploration.data_read import load_preprocessed
from helpers import train_split_by_column
from models.model_registry import (get_model_filename, register_model,
                                   discover_models, initialize_registry)


class ModelTrainer:
    def __init__(self, ws: int):
        self.ws = ws
        self.model_dir = Path("models") / f"ws_{ws}"
        self.initialize_systems()

        self.param_grids = {
            'SVM': {'C': [0.1, 1], 'kernel': ['linear']},
            'GLVQ': {'solver_type': ['adam', 'bfgs']},
            'HGBC': {'l2_regularization': [0.0, 0.2]},

        }

    def initialize_systems(self):
        initialize_registry()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.df = load_preprocessed(f'data/hole{self.ws}', keep_latlon=False, sample_frac=1)
        self.X_train, self.y_train, self.X_test, self.y_test = train_split_by_column(
            self.df, 'pothole', 0.9
        )
    @classmethod
    def get_feature_sets(cls, ws:int) -> List[List[str]]:
        with open("exploration/features/features_selected.json") as f:
            return list(json.load(f)[f'ws_{ws}'].values())

    def generate_configs(self) -> List[Dict]:
        configs = []
        for model_type, grid in self.param_grids.items():
            for params in product(*grid.values()):
                param_dict = dict(zip(grid.keys(), params))
                for features in self.get_feature_sets(self.ws):
                    configs.append({
                        'type': model_type,
                        'params': param_dict,
                        'features': features,
                        'ws': self.ws
                    })
        return configs

    def create_model(self, config: Dict):
        model_type = config['type']
        params = config['params']

        if model_type == 'GLVQ':
            return GLVQ(**params, prototype_n_per_class=[2, 2], random_state=42)
        elif model_type == 'HGBC':
            return HistGradientBoostingClassifier(**params, random_state=42)
        elif model_type == 'SVM':
            return SVC(**params, probability=True, random_state=42)
        raise ValueError(f"Unknown model: {model_type}")

    def train_model(self, config: Dict) -> Dict:
        model_path = get_model_filename(config['type'], config['params'], config['features'])
        print(model_path.name)
        # Check existing models - now properly returns metadata dict
        existing = discover_models(model_type=config['type'], params=config['params'])
        if existing:
            print(f"Using existing {config['type']} model")
            return existing[0]  # Returns the metadata dict from registry

        # Train new model
        model = self.create_model(config)
        model.fit(self.X_train[config['features']], self.y_train)

        # Evaluate
        pred = model.predict(self.X_test[config['features']])
        metrics = {
            'accuracy': accuracy_score(self.y_test, pred),
            'f1': f1_score(self.y_test, pred)
        }

        # Save model
        model_dir = self.model_dir / config['type']
        model_dir.mkdir(exist_ok=True)
        with open(model_dir / model_path.name, 'wb') as f:
            pickle.dump(model, f)

        # Create and return metadata dict
        model_info = {
            'config': config,
            'metrics': metrics,
            'path': str(model_dir / model_path.name),
            'model_object': model  # Optional: include actual model if needed
        }

        register_model(model_dir / model_path.name, metrics, config['features'])
        return model_info

    def run(self):
        self.load_data()
        best_score = 0
        best_model = None

        for config in self.generate_configs():
            # print(f"Training {config['type']} with {config['params']}")
            model_info = self.train_model(config)

            if model_info['metrics']['f1'] > best_score:
                best_score = model_info['metrics']['f1']
                best_model = model_info

        print(f"\nBest Model (F1: {best_score:.4f}):")
        print(f"Type: {best_model['config']['type']}")
        print(f"Params: {best_model['config']['params']}")
        print(f"Features: {len(best_model['config']['features'])}")


if __name__ == "__main__":
    for ws in [10]:
        print(f"\n=== Training WS={ws} ===")
        ModelTrainer(ws).run()