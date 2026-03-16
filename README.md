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
├── model.ipynb                          # Main notebook: dual-dataset pipeline (ML-Small + ML-25M)
├── model_1m.ipynb                       # Extended notebook: tri-dataset pipeline (+ ML-1M)
├── architecture_overview.pdf            # Architecture + data flow document for presentation
├── metric.txt                           # Summary of evaluation results across datasets
├── results_dual_dataset_comparison.txt  # Detailed results for ML-Small and ML-25M
├── results_tri_dataset_comparison.txt   # Detailed results for all three datasets
├── comparison_bar_chart.png             # Bar chart comparing key metrics
├── radar_chart.png                      # Radar chart for multi-metric comparison
├── coldstart_segment_f1.png             # F1 score chart for cold-start user segment
├── coldstart_segment_comparison.png     # Cold-start user comparison across datasets
├── sampled_eval_chart.png               # Sampled evaluation (Hit@K, NDCG@K) chart
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

## Results

| Metric | ML-Small | ML-25M | ML-1M |
| :--- | :---: | :---: | :---: |
| **RMSE** | 0.9264 | 1.1361 | 0.9674 |
| **MAE** | 0.7142 | 0.8714 | 0.7668 |
| **Precision** | 0.0635 | 0.0016 | 0.0925 |
| **Recall** | 0.0733 | 0.0302 | 0.0926 |
| **F1** | 0.0681 | 0.0031 | 0.0925 |

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

3. Open and run the notebooks in order:
   - `model.ipynb` — for ML-Small and ML-25M
   - `model_1m.ipynb` — for ML-Small, ML-25M, and ML-1M

---

## License

This project is for educational and research purposes.
