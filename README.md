# Hybrid Movie Recommendation System

A hybrid recommendation system that combines **Collaborative Filtering** (SVD), **Content-Based Filtering** (KNN), and **Deep Clustering** (IKSOM-CF + K-Means++) to generate high-quality, personalized movie recommendations. It includes adaptive cold-start handling and a validation-driven training process.

---

## Features

- **Improved Truncated SVD** for collaborative filtering with validation-based rank selection.
- **Content-Driven KNN** using 140-dimensional item feature vectors (genres, year, tags).
- **IKSOM-CF** (Improved Kohonen Self-Organizing Map) for deep item representation.
- **K-Means++ Clustering** with Silhouette Coefficient to find the optimal number of clusters.
- **Adaptive Alpha Blending** that dynamically shifts weight between CF and Content based on user rating density.
- **Cold-Start Fallback** using item popularity for users with no/few ratings.
- Evaluated on **three MovieLens datasets** — Small, 1M, and 25M.

---

## Project Structure

```
.
├── model.ipynb                          # Model 1: baseline using 140-dim sparse item features
├── model2.ipynb                         # Model 2: BERT-enhanced using 404-dim semantic embeddings
├── Result_model/                        # Evaluation charts and metrics for Model 1
│   ├── results_tri_dataset_comparison.txt
│   ├── comparison_bar_chart.png         
│   ├── radar_chart.png                 
│   └── sampled_eval_chart.png          
├── Result_model2/                       # Evaluation charts and metrics for Model 2
│   ├── results_tri_dataset_comparison.txt
│   ├── comparison_bar_chart.png         
│   ├── radar_chart.png                 
│   └── sampled_eval_chart.png          
└── data/                                # (Not tracked) Raw dataset directories
    ├── ml-small/
    ├── ml-1m/
    └── ml-25m/
```

---

## Architecture Overview

The pipeline processes data through 4 main stages:

1. **Data Preparation** — Splits ratings into 70% train / 15% val / 15% test. Extracts 140-dim item features.
2. **Collaborative Filtering (SVD)** — Factorizes the user-item matrix. Best SVD rank `k` is chosen by validation RMSE to prevent test leakage.
3. **Content-Based + Clustering** — Content-Driven KNN scores items by feature similarity. IKSOM-CF deep clusters all items via K-Means++.
4. **Hybrid Scoring** — Adaptively blends CF and content scores using Alpha weighting; filters candidates by the user's preferred cluster; re-ranks by SVD score.

See `architecture_overview.pdf` for a full explanation with diagrams.

---

## Datasets

Download the datasets from [MovieLens](https://grouplens.org/datasets/movielens/) and place them in `./data/`:

| Dataset | Directory | Ratings |
| :--- | :--- | :--- |
| MovieLens Small | `data/ml-small` | ~100K |
| MovieLens 1M | `data/ml-1m` | ~1M |
| MovieLens 25M | `data/ml-25m` | ~25M |

> **Note:** The `ml-1m` dataset uses `.dat` files (`::`-separated). The pipeline handles this automatically.

---

## Model Variants & Performance

We evaluate two variants of the system based on their item feature representations:

1. **Model 1 (`model.ipynb`)**: Uses a **140-dim sparse vector** (one-hot genres + year bucket + binary top-100 tags).
2. **Model 2 (`model2.ipynb`)**: Uses a **404-dim dense semantic embedding** (`all-MiniLM-L6-v2` BERT encoding + year bucket). 

### Performance Comparison

#### MovieLens Small (ML-Small)
| Metric | Model 1 (Sparse) | Model 2 (BERT) | % Difference |
| :--- | :---: | :---: | :---: |
| **RMSE** | 0.9264 | 0.9264 | 0.00% |
| **MAE** | 0.7142 | 0.7142 | 0.00% |
| **Precision** | 0.0453 | 0.0644 | **+42.16%** 🟢 |
| **Recall** | 0.0511 | 0.0781 | **+52.84%** 🟢 |
| **F1** | 0.0480 | 0.0706 | **+47.08%** 🟢 |
| **MRR** | 0.2828 | 0.2827 | -0.04% 🔴 |
| **Hit@10** | 0.4621 | 0.4622 | +0.02% 🟢 |
| **NDCG@10** | 0.3126 | 0.3125 | -0.03% 🔴 |
| **Hit@20** | 0.5376 | 0.5391 | +0.28% 🟢 |
| **NDCG@20** | 0.3318 | 0.3320 | +0.06% 🟢 |

#### MovieLens 1M (ML-1M)
| Metric | Model 1 (Sparse) | Model 2 (BERT) | % Difference |
| :--- | :---: | :---: | :---: |
| **RMSE** | 0.9674 | 0.9674 | 0.00% |
| **MAE** | 0.7668 | 0.7668 | 0.00% |
| **Precision** | 0.0982 | 0.0843 | -14.15% 🔴 |
| **Recall** | 0.0986 | 0.0841 | -14.71% 🔴 |
| **F1** | 0.0984 | 0.0842 | -14.43% 🔴 |
| **MRR** | 0.2987 | 0.2978 | -0.30% 🔴 |
| **Hit@10** | 0.5071 | 0.5063 | -0.16% 🔴 |
| **NDCG@10** | 0.3367 | 0.3358 | -0.27% 🔴 |
| **Hit@20** | 0.5764 | 0.5757 | -0.12% 🔴 |
| **NDCG@20** | 0.3544 | 0.3535 | -0.25% 🔴 |

#### MovieLens 25M (ML-25M)
| Metric | Model 1 (Sparse) | Model 2 (BERT) | % Difference |
| :--- | :---: | :---: | :---: |
| **RMSE** | 0.8964 | 0.8964 | 0.00% |
| **MAE** | 0.6851 | 0.6851 | 0.00% |
| **Precision** | 0.0933 | 0.0823 | -11.79% 🔴 |
| **Recall** | 0.1089 | 0.1016 | -6.70% 🔴 |
| **F1** | 0.1005 | 0.0909 | -9.55% 🔴 |
| **MRR** | 0.4212 | 0.4236 | +0.57% 🟢 |
| **Hit@10** | 0.5480 | 0.5490 | +0.18% 🟢 |
| **NDCG@10** | 0.4437 | 0.4458 | +0.47% 🟢 |
| **Hit@20** | 0.5735 | 0.5728 | -0.12% 🔴 |
| **NDCG@20** | 0.4502 | 0.4519 | +0.38% 🟢 |

*Note: Model 2 (BERT-enhanced) shows substantial improvements in Precision, Recall, and F1 on the ML-Small dataset (up to +52%) because the semantic features help mitigate data sparsity. Model 1 (Sparse features) performs slightly better structurally on the older and larger subsets (ML-1M, ML-25M).*

---

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/anandiitp/Hybrid-Movie-Recommendation-System.git
   cd Hybrid-Movie-Recommendation-System
   ```

2. Place the datasets in the `data/` directory (see above).

3. Open and run the notebooks:
   - `model.ipynb` — to run the pipeline with hand-crafted features
   - `model2.ipynb` — to run the pipeline with BERT semantic embeddings

