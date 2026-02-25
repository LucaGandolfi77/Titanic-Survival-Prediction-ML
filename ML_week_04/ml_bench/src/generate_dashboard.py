#!/usr/bin/env python3
"""
Generate interactive HTML dashboard from results.

Usage:
    python src/generate_dashboard.py --results results/ --output reports/dashboard.html
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard import generate_dashboard

def main():
    parser = argparse.ArgumentParser(description='Generate dashboard')
    parser.add_argument('--results', type=str, nargs='+', default=['results/'],
                       help='Results directory or directories')
    parser.add_argument('--output', type=str, default='outputs/dashboard.html',
                       help='Output HTML file')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Comma-separated metrics to include')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 Dashboard Generator")
    print("=" * 60)
    
    # Collect all results
    all_results = {}
    
    for results_path in args.results:
        results_dir = Path(results_path)
        if not results_dir.exists():
            print(f"⚠️  Results dir not found: {results_path}")
            continue
        
        # Find all metrics.json files
        for metrics_file in results_dir.glob('*/metrics.json'):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    all_results.update(data)
                    print(f"✅ Loaded {len(data)} models from {metrics_file}")
            except Exception as e:
                print(f"❌ Error loading {metrics_file}: {e}")
    
    if not all_results:
        print("⚠️  No results found. Generating empty dashboard...")
        all_results = {"placeholder": {"accuracy": 0.0}}
    
    # Generate dashboard
    output_file = Path(args.output)
    generate_dashboard(all_results, str(output_file))
    print(f"\n✅ Dashboard: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
