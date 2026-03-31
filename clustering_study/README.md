# Clustering Adattivo — Confronto tra K-Means, ISODATA e Metodi di Validazione

Studio sperimentale sul clustering come *processo di scoperta della struttura dei dati*,
con attenzione alla selezione del numero di cluster, alla rappresentazione delle
partizioni e alla validità senza ground-truth forte.

## Domande di Ricerca

| # | Domanda |
|---|---------|
| RQ1 | Qual è l'impatto della geometria dei dati sulle prestazioni di K-Means, ISODATA e GMM? |
| RQ2 | Come si confrontano Elbow, Silhouette e Gap Statistic nella determinazione di *k*? |
| RQ3 | Un framework adattivo split/merge può eguagliare o superare metodi con *k* oracolo? |
| RQ4 | Quanto sono stabili le partizioni rispetto all'inizializzazione (multi-seed)? |
| RQ5 | Come degradano le prestazioni al crescere della dimensionalità e del rumore? |

## Algoritmi

| Metodo | Tipo | K fisso? |
|--------|------|----------|
| K-Means (k-means++) | Centroide | Sì |
| Mini-Batch K-Means | Centroide (scalabile) | Sì |
| Bisecting K-Means | Divisivo gerarchico | Sì |
| GMM (EM) | Modello probabilistico | Sì |
| ISODATA | Centroide + split/merge/discard | No — adattivo |
| Adaptive Split/Merge | Framework dinamico basato su silhouette e distanza centroidi | No — adattivo |

## Metriche di Validazione

### Indici Interni (senza ground-truth)

- **Silhouette Score** — coesione vs. separazione per campione
- **Calinski-Harabasz** — rapporto varianza inter/intra
- **Davies-Bouldin** — media delle similarità tra cluster (più basso = meglio)
- **Dunn Index** — min distanza inter-cluster / max diametro intra-cluster
- **WCSS** — somma dei quadrati intra-cluster
- **BCSS** — somma dei quadrati inter-cluster

### Indici Esterni (con ground-truth)

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Fowlkes-Mallows Index (FMI)
- Homogeneity, Completeness, V-measure

### Selezione di K

- Metodo del gomito (Elbow) con `kneed`
- Silhouette massima
- Gap Statistic (Tibshirani et al., 2001)

### Stabilità

- Bootstrap ARI su 15 semi casuali

## Dataset

### Sintetici (6 geometrie)

| Nome | Descrizione |
|------|-------------|
| `blobs` | Cluster gaussiani ben separati |
| `moons` | Due mezzelune intrecciate |
| `circles` | Cerchi concentrici |
| `anisotropic` | Cluster con trasformazione lineare |
| `varied_variance` | Cluster con dispersioni diverse |
| `unbalanced` | Cluster con numerosità diverse |

### Reali (3 benchmark)

Iris, Wine, Breast Cancer (da `sklearn.datasets`).

## Esperimenti

| # | Esperimento | Descrizione |
|---|-------------|-------------|
| 1 | K Selection | Tutti i dataset × k ∈ {2..15} × 15 semi × 6 metodi |
| 2 | Geometry | Impatto della geometria sui 6 dataset sintetici |
| 3 | Initialization | Sensitività all'inizializzazione (ARI a coppie tra semi) |
| 4 | Scalability | Tempo di fit vs. dimensione del dataset |
| 5 | Adaptive | Metodi adattivi (ISODATA, Adaptive) vs. oracolo con k ottimo |
| 6 | Noise | Robustezza al rumore gaussiano e agli outlier |
| 7 | High-Dim | Comportamento al crescere della dimensionalità (2→100 feature) |

## Quick Start

```bash
cd clustering_study
pip install -r requirements.txt

# Dry-run (veloce, ~30s)
python -m experiments.run_all --dry-run

# Esecuzione completa
python -m experiments.run_all

# Test
python -m pytest tests/ -v
```

## Struttura del Progetto

```
clustering_study/
├── config.py                       # Configurazione centrale (dataclass frozen)
├── requirements.txt
├── README.md
│
├── data/
│   ├── __init__.py
│   ├── synthetic.py                # 6 generatori sintetici
│   ├── real_datasets.py            # iris, wine, breast_cancer
│   ├── preprocessor.py             # StandardScaler, PCA
│   └── noise_injector.py           # Rumore e outlier
│
├── algorithms/
│   ├── __init__.py
│   ├── kmeans.py                   # K-Means + storia centroidi
│   ├── isodata.py                  # ISODATA completo (Ball & Hall, 1965)
│   ├── minibatch_kmeans.py         # Mini-Batch K-Means
│   ├── bisecting_kmeans.py         # Bisecting K-Means
│   ├── gmm.py                      # GMM (Gaussian Mixture)
│   ├── adaptive_clustering.py      # Framework adattivo split/merge
│   └── algorithm_factory.py        # Factory per tutti i metodi
│
├── validation/
│   ├── __init__.py
│   ├── internal_indices.py         # Silhouette, CH, DB, Dunn, WCSS, BCSS
│   ├── external_indices.py         # ARI, NMI, FMI, V-measure
│   ├── gap_statistic.py            # Gap Statistic (Tibshirani 2001)
│   ├── stability.py                # Bootstrap ARI
│   └── k_selection.py              # Elbow, Silhouette, Gap
│
├── experiments/
│   ├── __init__.py
│   ├── utils.py                    # evaluate_clustering()
│   ├── exp_k_selection.py          # Esperimento 1
│   ├── exp_geometry.py             # Esperimento 2
│   ├── exp_initialization.py       # Esperimento 3
│   ├── exp_scalability.py          # Esperimento 4
│   ├── exp_adaptive.py             # Esperimento 5
│   ├── exp_noise_robustness.py     # Esperimento 6
│   ├── exp_high_dimensional.py     # Esperimento 7
│   └── run_all.py                  # Orchestratore con --dry-run
│
├── visualization/
│   ├── __init__.py
│   ├── _common.py                  # Palette, salvataggio
│   ├── cluster_scatter.py          # Scatter 2D
│   ├── elbow_plot.py               # Curva del gomito
│   ├── silhouette_plot.py          # Diagramma silhouette
│   ├── gap_plot.py                 # Gap statistic
│   ├── stability_heatmap.py        # Heatmap stabilità
│   ├── split_merge_evolution.py    # Evoluzione k per ISODATA/Adaptive
│   ├── validation_comparison.py    # Confronto metriche
│   ├── centroid_evolution.py       # Movimento centroidi
│   ├── pca_projection.py           # Proiezione PCA 2D/3D
│   └── scalability_plot.py         # Tempo vs dimensione
│
├── evaluation/
│   ├── __init__.py
│   ├── statistical_tests.py        # Friedman + post-hoc, Wilcoxon
│   └── report_generator.py         # Report Markdown
│
└── tests/
    ├── test_isodata.py             # 13 test ISODATA
    ├── test_adaptive.py            # 11 test Adaptive
    ├── test_validation.py          # 11 test indici interni/esterni
    ├── test_stability.py           # 12 test stabilità + selezione k
    └── test_algorithm_factory.py   # 7 test factory
```

## Riferimenti

- Ball, G.H. & Hall, D.J. (1965). *ISODATA, a novel method of data analysis and pattern classification*. Stanford Research Institute.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). *Estimating the number of clusters in a data set via the gap statistic*. JRSS Series B, 63(2), 411–423.
- Rousseeuw, P.J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. J. Computational and Applied Mathematics, 20, 53–65.
- Caliński, T. & Harabasz, J. (1974). *A dendrite method for cluster analysis*. Communications in Statistics, 3(1), 1–27.
- Davies, D.L. & Bouldin, D.W. (1979). *A cluster separation measure*. IEEE PAMI, 1(2), 224–227.
