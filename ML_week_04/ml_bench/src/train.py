#!/usr/bin/env python3
"""
Train ML models on selected dataset.

Usage:
    python src/train.py --config config/experiment.yaml --output results/
    python src/train.py --dataset iris --algorithms logistic_regression,random_forest
"""

import argparse
import time
import json
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.models import MLModel
from src.evaluation import compute_classification_metrics, compute_regression_metrics
from src.utils import load_config, save_config

def main():
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--config', type=str, default='config/experiment.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Override dataset name')
    parser.add_argument('--algorithms', type=str, default=None,
                       help='Comma-separated algorithm names')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"⚠️  Config not found: {args.config}")
        config = {
            'dataset': {'name': 'iris', 'train_size': 0.8, 'random_state': 42},
            'algorithms': [
                {'name': 'logistic_regression', 'params': {'C': 1.0, 'max_iter': 1000}},
                {'name': 'random_forest', 'params': {'n_estimators': 100, 'max_depth': 10}},
            ]
        }
    
    # Override if provided
    if args.dataset:
        config['dataset']['name'] = args.dataset
    
    dataset_name = config['dataset']['name']
    output_dir = Path(args.output) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 ML Training Pipeline")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Output:  {output_dir}")
    
    # Load data
    print(f"\n📦 Loading {dataset_name}...")
    X_train, X_test, y_train, y_test, task_type = load_dataset(
        dataset_name,
        train_size=config['dataset'].get('train_size', 0.8),
        random_state=config['dataset'].get('random_state', 42)
    )
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}, Task: {task_type}")
    
    # Train models
    results = {}
    algorithms = config.get('algorithms', [])
    
    if args.algorithms:
        algo_names = args.algorithms.split(',')
        algorithms = [a for a in algorithms if a['name'] in algo_names]
    
    for algo_config in algorithms:
        algo_name = algo_config['name']
        params = algo_config.get('params', {})
        
        print(f"\n🔧 Training {algo_name}...")
        start_time = time.time()
        
        try:
            model = MLModel(name=algo_name, model_type=algo_name, params=params)
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            if task_type == "classification":
                y_pred_proba = model.predict_proba(X_test)
                metrics = compute_classification_metrics(y_test, y_pred, y_pred_proba)
            else:
                metrics = compute_regression_metrics(y_test, y_pred)
            
            metrics['train_time'] = train_time
            results[algo_name] = metrics
            
            print(f"✅ {algo_name:25} | Accuracy: {metrics.get('accuracy', metrics.get('mse', 0)):.4f} | Time: {train_time:.2f}s")
        
        except Exception as e:
            print(f"❌ {algo_name:25} | Error: {str(e)[:50]}")
            results[algo_name] = {'error': str(e)}
    
    # Save results
    results_file = output_dir / 'metrics.json'
    with open(results_file, 'w') as f:
        # Convert non-serializable objects
        clean_results = {}
        for name, metrics in results.items():
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    clean_metrics[k] = v
            clean_results[name] = clean_metrics
        json.dump(clean_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print("=" * 60)
    print("✨ Training complete!")

if __name__ == "__main__":
    main()
