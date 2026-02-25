# ML Benchmarking & Performance Dashboard

A comprehensive machine learning project that enables **dataset selection**, **algorithm comparison**, and **interactive HTML dashboards** to visualize performance metrics, training speed, and feature importance across different ML methods.

## Overview

This project provides a flexible framework for:
- 🎯 **Multi-dataset support**: Choose from famous ML datasets (Iris, Wine, Breast Cancer, MNIST, Fashion-MNIST, California Housing, Adult Income, Titanic)
- ⚡ **Algorithm comparison**: Compare classical ML methods (Logistic Regression, Random Forest, SVM, XGBoost, Neural Networks)
- 📊 **Interactive dashboards**: Generate HTML pages with real-time performance metrics, confusion matrices, and latency comparisons
- 🔍 **Feature analysis**: Study impact of features, dimensionality reduction (PCA, t-SNE), and scaling methods
- 📈 **Benchmarking**: Track training time, inference speed, memory usage, and model complexity

## Quick Start

### Installation

```bash
# Clone or navigate to the project
cd ML_week_04

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config/experiment.yaml` to select your dataset and algorithms:

```yaml
dataset:
  name: "iris"  # or: wine, breast_cancer, mnist, housing, adult, titanic
  train_size: 0.8
  random_state: 42
  
algorithms:
  - name: "logistic_regression"
    params: {C: 1.0, max_iter: 1000}
  - name: "random_forest"
    params: {n_estimators: 100, max_depth: 10}
  - name: "svm"
    params: {C: 1.0, kernel: "rbf"}
  - name: "xgboost"
    params: {n_estimators: 100, learning_rate: 0.1}
  - name: "neural_network"
    params: {hidden_layer_sizes: (128, 64), max_iter: 500}

features:
  scale: true  # Standardize features
  reduction: null  # or: "pca", "tsne"
  n_components: 2
  
benchmark:
  n_runs: 5  # Average timing over N runs
  measure_memory: true
  compute_shap: true  # Feature importance via SHAP
```

### Run Experiments

```bash
# Train all algorithms on selected dataset
python src/train.py --config config/experiment.yaml --output results/

# Run benchmarking suite
python src/benchmark.py --config config/experiment.yaml --output results/benchmarks/

# Generate interactive HTML dashboard
python src/generate_dashboard.py --results results/ --output reports/dashboard.html
```

## Project Structure

```
ML_week_04/
├── config/
│   ├── experiment.yaml           # Main experiment configuration
│   ├── datasets.yaml             # Dataset-specific settings
│   └── algorithms.yaml           # Algorithm hyperparameters
├── src/
│   ├── data/
│   │   ├── loaders.py            # Dataset loading & preprocessing
│   │   ├── preprocessing.py      # Scaling, PCA, feature engineering
│   │   └── splitters.py          # Train/test/validation splits
│   ├── models/
│   │   ├── classical.py          # Classical ML models
│   │   ├── nn.py                 # Neural network implementations
│   │   └── ensemble.py           # Ensemble methods
│   ├── training/
│   │   ├── trainer.py            # Single model trainer
│   │   ├── validator.py          # Cross-validation & metrics
│   │   └── callbacks.py          # Early stopping, checkpoints
│   ├── evaluation/
│   │   ├── metrics.py            # Performance metrics
│   │   ├── visualization.py      # Plots: confusion matrix, ROC, learning curves
│   │   └── feature_importance.py # SHAP, permutation importance
│   ├── benchmarking/
│   │   ├── profiler.py           # Timing & memory profiling
│   │   ├── latency.py            # Inference latency analysis
│   │   └── comparator.py         # Cross-model comparison
│   ├── dashboard/
│   │   ├── html_generator.py     # Interactive HTML templates
│   │   ├── plotly_charts.py      # Plotly interactive visualizations
│   │   └── report_builder.py     # Aggregate results into reports
│   ├── utils/
│   │   ├── config.py             # Config loading & validation
│   │   ├── logging.py            # Structured logging
│   │   └── helpers.py            # Utility functions
│   ├── train.py                  # Main training CLI
│   ├── benchmark.py              # Benchmarking CLI
│   └── generate_dashboard.py     # Dashboard generation CLI
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline_models.ipynb   # Quick baseline comparison
│   ├── 03_hyperparameter_tuning.ipynb  # Hyperparameter optimization
│   └── 04_advanced_analysis.ipynb # Advanced feature engineering
├── tests/
│   ├── test_loaders.py           # Dataset loading tests
│   ├── test_models.py            # Model instantiation tests
│   └── test_metrics.py           # Metrics computation tests
├── outputs/
│   ├── models/                   # Trained model checkpoints
│   ├── plots/                    # Generated plots & figures
│   └── dashboards/               # Interactive HTML reports
├── requirements.txt              # Python dependencies
├── Makefile                      # Convenient commands
└── README.md                     # This file
```

## Supported Datasets

| Dataset | Type | Samples | Features | Classes | Task |
|---------|------|---------|----------|---------|------|
| **Iris** | Classification | 150 | 4 | 3 | Multi-class |
| **Wine** | Classification | 178 | 13 | 3 | Multi-class |
| **Breast Cancer** | Classification | 569 | 30 | 2 | Binary |
| **MNIST** | Classification | 70,000 | 784 | 10 | Multi-class (digits) |
| **Fashion-MNIST** | Classification | 60,000 | 784 | 10 | Multi-class (clothing) |
| **California Housing** | Regression | 20,640 | 8 | — | Continuous |
| **Adult Income** | Classification | 48,842 | 14 | 2 | Binary (>50K income) |
| **Titanic** | Classification | 891 | 7–10* | 2 | Binary (survival) |

*Titanic features are post-preprocessing (encoding, missing value imputation)

### Load Custom Datasets

Add a custom loader in `src/data/loaders.py`:

```python
def load_custom_dataset(dataset_name: str, **kwargs):
    if dataset_name == "my_dataset":
        X, y = load_from_csv("path/to/data.csv")
        return X, y, "classification"
    raise ValueError(f"Unknown dataset: {dataset_name}")
```

Update `config/datasets.yaml`:
```yaml
my_dataset:
  url: "path/to/data.csv"
  task_type: "classification"
  target_col: "label"
```

## Supported Algorithms

### Classical Methods
- **Logistic Regression**: Fast baseline for classification
- **Random Forest**: Ensemble method, robust to non-linearity
- **Support Vector Machine (SVM)**: Kernel-based classifier
- **XGBoost / LightGBM**: Gradient boosting, excellent accuracy
- **k-Nearest Neighbors**: Non-parametric classifier

### Deep Learning
- **Multi-layer Perceptron (MLP)**: PyTorch-based feed-forward network
- **Convolutional Neural Network (CNN)**: For image-like data (MNIST, Fashion-MNIST)

### Ensemble Methods
- **Voting Classifier**: Soft/hard voting across multiple models
- **Stacking**: Meta-learner on top of base learners

## Key Features

### 1. **Performance Metrics**
Automatically computed for each model:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Cross-validation scores
- Training time, inference time
- Memory footprint

### 2. **Interactive Dashboard**
Generated HTML dashboards include:
- **Model Comparison Panel**: Side-by-side metrics comparison
- **Confusion Matrices**: Heatmaps for each model
- **ROC Curves**: Interactive multi-model ROC visualization
- **Learning Curves**: Training/validation loss over epochs
- **Feature Importance**: Top N features via SHAP or permutation
- **Latency Heatmap**: Training vs. inference time across models
- **Dataset Overview**: Histograms, correlations, missing values
- **Export & Download**: Save plots as PNG/SVG

### 3. **Benchmarking Suite**
Track:
- **Training time**: Total time to fit model
- **Inference latency**: Single prediction + batch prediction
- **Memory peak**: Maximum RAM usage during training/inference
- **Throughput**: Samples/second for batch inference
- **Scalability**: Performance vs. dataset size

### 4. **Feature Analysis**
- **Correlation matrix**: Identify feature relationships
- **Feature importance**: SHAP values, permutation importance, MDI
- **Dimensionality reduction**: Visualize PCA/t-SNE projections
- **Scaling comparison**: Impact of StandardScaler, MinMaxScaler, RobustScaler

## Usage Examples

### Example 1: Quick Iris Classification

```bash
# Use pre-configured Iris setup
python src/train.py --config config/presets/iris_quick.yaml

# Generate dashboard
python src/generate_dashboard.py --results results/iris/ --output reports/iris_dashboard.html
```

### Example 2: MNIST with Multiple Algorithms

```yaml
# config/experiment.yaml
dataset:
  name: "mnist"
  train_size: 0.8

algorithms:
  - name: "random_forest"
    params: {n_estimators: 50, max_depth: 15}
  - name: "svm"
    params: {kernel: "rbf", C: 10}
  - name: "cnn"
    params: {conv_layers: 2, dense_layers: 2}
```

```bash
python src/train.py --config config/experiment.yaml --output results/mnist_comparison/
python src/benchmark.py --config config/experiment.yaml --output results/mnist_benchmark/
python src/generate_dashboard.py --results results/ --output reports/mnist_analysis.html
```

### Example 3: Compare Scaling Methods

```bash
# Run same experiment with different scaling strategies
for scaler in StandardScaler MinMaxScaler RobustScaler; do
  python src/train.py \
    --config config/experiment.yaml \
    --scaling $scaler \
    --output results/scaling_$scaler/
done

python src/generate_dashboard.py \
  --results results/scaling_* \
  --output reports/scaling_comparison.html
```

## Commands Reference

### Training

```bash
# Train with default config
python src/train.py

# Train with custom config
python src/train.py --config my_config.yaml

# Specify output directory
python src/train.py --output ./my_results/

# Override dataset
python src/train.py --dataset breast_cancer

# Override algorithms
python src/train.py --algorithms logistic_regression,random_forest,svm

# Enable verbose logging
python src/train.py --verbose
```

### Benchmarking

```bash
# Run full benchmarking suite
python src/benchmark.py --config config/experiment.yaml

# Benchmark specific algorithms
python src/benchmark.py --algorithms xgboost,neural_network

# Set number of timing runs
python src/benchmark.py --n_runs 10

# Include memory profiling
python src/benchmark.py --measure_memory
```

### Dashboard Generation

```bash
# Generate dashboard from results directory
python src/generate_dashboard.py --results results/ --output dashboard.html

# Include specific metrics
python src/generate_dashboard.py --results results/ --metrics accuracy,f1,latency

# Generate for comparison (multiple result folders)
python src/generate_dashboard.py --results results/model_a results/model_b --output comparison.html
```

### Makefile Commands

```bash
make install              # Install dependencies
make train                # Run default experiment
make benchmark            # Run benchmarking
make dashboard            # Generate dashboard
make test                 # Run unit tests
make clean                # Clean outputs
make help                 # Show all commands
```

## Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
shap>=0.45.0
pyyaml>=6.0
click>=8.1.0
tqdm>=4.66.0
pytest>=7.4.0
```

Install all with:
```bash
pip install -r requirements.txt
```

## Configuration Deep Dive

### Dataset Configuration (config/datasets.yaml)

```yaml
iris:
  source: "sklearn"
  task_type: "classification"
  n_classes: 3
  features: 4
  
mnist:
  source: "torchvision"
  task_type: "classification"
  image_size: [28, 28]
  n_classes: 10

breast_cancer:
  source: "sklearn"
  task_type: "classification"
  preprocessing:
    handle_missing: "mean"
    scale: "standardscaler"
```

### Algorithm Hyperparameters (config/algorithms.yaml)

```yaml
logistic_regression:
  default: {C: 1.0, solver: "lbfgs", max_iter: 1000}
  tuning_grid: {C: [0.1, 1, 10]}

random_forest:
  default: {n_estimators: 100, max_depth: 10}
  tuning_grid: {n_estimators: [50, 100, 200], max_depth: [5, 10, 20]}

neural_network:
  default: {hidden_layers: [128, 64], activation: "relu", learning_rate: 0.001}
  tuning_grid: {hidden_layers: [[64], [128, 64], [256, 128, 64]]}
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Load dataset, visualize distributions, correlations, missing values |
| `02_baseline_models.ipynb` | Quick training of baseline models, initial metric comparison |
| `03_hyperparameter_tuning.ipynb` | GridSearch / RandomSearch for optimal hyperparameters |
| `04_advanced_analysis.ipynb` | SHAP analysis, feature interactions, learning curves, ablation studies |

## Dashboard Features

The generated HTML dashboards include:

### Tabs
1. **Overview**: Dataset summary, class distribution, feature statistics
2. **Model Performance**: Accuracy, precision, recall, F1 across models
3. **Confusion Matrices**: Interactive heatmaps for each classifier
4. **ROC & PR Curves**: Multi-model comparison with AUC scores
5. **Learning Curves**: Training/validation loss, accuracy over epochs
6. **Feature Importance**: SHAP beeswarm/summary plots
7. **Benchmarking**: Training time, inference latency, memory usage
8. **Predictions**: Sample predictions, prediction uncertainty
9. **Comparison**: Side-by-side metrics export (CSV)

### Interactive Elements
- ✅ Hover tooltips with exact values
- ✅ Zoom & pan on plots
- ✅ Toggle series visibility (legend click)
- ✅ Download plots as PNG
- ✅ Filter models by performance threshold
- ✅ Sort metrics table by column

## Advanced Usage

### Hyperparameter Tuning

```bash
python src/hyperparameter_search.py \
  --dataset iris \
  --algorithm random_forest \
  --method gridserach \
  --param_grid "n_estimators=[50,100,200];max_depth=[5,10,20]"
```

### Cross-Validation

Edit `config/experiment.yaml`:
```yaml
validation:
  method: "cross_validation"  # or: "holdout", "stratified_kfold"
  n_splits: 5
  random_state: 42
```

### SHAP Feature Analysis

```python
python src/feature_importance.py \
  --model outputs/models/random_forest.pkl \
  --dataset iris \
  --method shap \
  --output outputs/shap_analysis.html
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
# or for CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Dashboard generation is slow
- Reduce `n_runs` in benchmark config
- Disable `compute_shap: false`
- Use smaller dataset or subset for prototyping

### Issue: MNIST model out of memory
- Reduce batch size in `config/algorithms.yaml`
- Use CNN instead of fully-connected network
- Reduce dataset size (train_size: 0.5)

## Citation & References

If you use this project, cite the dataset and algorithm papers:

1. Iris: Fisher, R.A. (1936)
2. MNIST: LeCun et al. (1998)
3. XGBoost: Chen & Guestrin (2016)
4. SHAP: Lundberg & Lee (2017)
5. PyTorch: Paszke et al. (2019)

## License

MIT License — See LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-dataset`)
3. Commit changes (`git commit -m "Add XYZ dataset support"`)
4. Push to branch (`git push origin feature/new-dataset`)
5. Open Pull Request

## Authors

Created as part of ML Week 04 benchmarking & performance analysis project.

## Changelog

### v1.0.0 (2026-02-24)
- ✅ Multi-dataset support (8 datasets)
- ✅ 5+ classical ML algorithms
- ✅ Interactive HTML dashboards
- ✅ Benchmarking suite with timing/memory profiling
- ✅ SHAP feature importance analysis
- ✅ Comprehensive configuration system
- ✅ Jupyter notebooks for exploration

## Support

For issues, questions, or feature requests:
- 📧 Open an Issue on GitHub
- 💬 Check existing discussions
- 📖 Read the docs in `docs/` (if available)

---

**Happy ML Benchmarking! 🚀**
