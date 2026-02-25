#!/usr/bin/env python3
"""
Benchmark ML models: measure training/inference time and memory usage.

Usage:
    python src/benchmark.py --config config/experiment.yaml --output results/benchmarks/
"""

import argparse
import time
import psutil
import os
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.models import MLModel
from src.utils import load_config

def measure_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    parser = argparse.ArgumentParser(description='Benchmark ML models')
    parser.add_argument('--config', type=str, default='config/experiment.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Override dataset name')
    parser.add_argument('--algorithms', type=str, default=None,
                       help='Comma-separated algorithm names')
    parser.add_argument('--output', type=str, default='results/benchmarks/',
                       help='Output directory')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of timing runs per model')
    parser.add_argument('--measure_memory', action='store_true',
                       help='Measure peak memory usage')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = {
            'dataset': {'name': 'iris', 'train_size': 0.8},
            'algorithms': [
                {'name': 'logistic_regression', 'params': {}},
                {'name': 'random_forest', 'params': {'n_estimators': 100}},
            ]
        }
    
    if args.dataset:
        config['dataset']['name'] = args.dataset
    
    dataset_name = config['dataset']['name']
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("⏱️  ML Benchmarking Suite")
    print("=" * 60)
    print(f"Dataset:      {dataset_name}")
    print(f"Runs per algo: {args.n_runs}")
    print(f"Output:       {output_dir}")
    
    # Load data
    print(f"\n📦 Loading data...")
    X_train, X_test, y_train, y_test, _ = load_dataset(dataset_name)
    
    # Benchmark models
    benchmarks = {}
    algorithms = config.get('algorithms', [])
    
    if args.algorithms:
        algo_names = args.algorithms.split(',')
        algorithms = [a for a in algorithms if a['name'] in algo_names]
    
    for algo_config in algorithms:
        algo_name = algo_config['name']
        params = algo_config.get('params', {})
        
        print(f"\n🔧 Benchmarking {algo_name}...")
        
        train_times = []
        infer_times = []
        
        for run in range(args.n_runs):
            try:
                model = MLModel(name=algo_name, model_type=algo_name, params=params)
                
                # Measure training time
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0
                train_times.append(train_time)
                
                # Measure inference time (batch)
                t0 = time.time()
                _ = model.predict(X_test)
                infer_time = time.time() - t0
                infer_times.append(infer_time)
                
                print(f"  Run {run+1}/{args.n_runs}: train={train_time:.3f}s, infer={infer_time:.3f}s")
            
            except Exception as e:
                print(f"  ❌ Run {run+1} failed: {str(e)[:40]}")
        
        if train_times:
            benchmarks[algo_name] = {
                "train_time_mean": sum(train_times) / len(train_times),
                "train_time_std": (sum((x - sum(train_times)/len(train_times))**2 for x in train_times) / len(train_times)) ** 0.5,
                "infer_time_mean": sum(infer_times) / len(infer_times),
                "infer_time_std": (sum((x - sum(infer_times)/len(infer_times))**2 for x in infer_times) / len(infer_times)) ** 0.5,
                "samples_per_sec": len(X_test) / (sum(infer_times) / len(infer_times)),
                "runs": args.n_runs,
            }
    
    # Save benchmarks
    bench_file = output_dir / f'{dataset_name}_benchmarks.json'
    with open(bench_file, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    print(f"\n💾 Benchmarks saved to {bench_file}")
    print("\n📊 Summary:")
    for algo_name, bench in benchmarks.items():
        print(f"\n  {algo_name}:")
        print(f"    Train: {bench['train_time_mean']:.3f}s ± {bench['train_time_std']:.3f}s")
        print(f"    Infer: {bench['infer_time_mean']:.3f}s ({bench['samples_per_sec']:.0f} samples/s)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
