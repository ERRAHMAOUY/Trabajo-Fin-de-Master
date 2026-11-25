# Trabajo-Fin-de-Master
# Ray vs. Apache Spark for Machine Learning Workflows on Large Datasets

**Master's Thesis (TFM)** · Master's Degree in Computer Engineering, Engineering and Data Science
**Universidad de Málaga** · E.T.S. de Ingeniería Informática · May 2025

- **Author:** Brahim Er Rahmaouy
- **Supervisor:** José Manuel García Nieto
- **Co-supervisor:** Antonio Jesús Nebro Urbaneja

> 📄 Full report: [`docs/Memory_Report.pdf`](docs/Memory_Report.pdf) (English, with Spanish abstract and conclusions summary)

---

## Overview

This project compares **Ray** and **Apache Spark** as platforms for distributed machine learning. The same ML workflow is implemented in both and benchmarked on a large real-world dataset, measuring:

- Execution time (feature processing, training, prediction)
- Resource utilization (CPU and memory)
- Scalability as the dataset grows
- Prediction throughput under different batch sizes
- Implementation effort and developer experience

### Research questions

1. Which platform offers better performance for ML tasks?
2. What are the advantages and disadvantages of each in terms of scalability and ease of use?

## Dataset

**Chicago "Crimes – 2001 to Present"** (Chicago Police Department CLEAR system, snapshot `Crimes_-_2001_to_Present_20250127.csv`).

- ~8.31 million rows, 22 columns
- **Task:** binary classification, predicting whether an **arrest** was made
- Tested at three sizes: **1M** (random subset), **4M** (random subset) and **Full (8.31M)**

The dataset is not included in this repository because of its size. Download it from the [Chicago Data Portal](https://data.cityofchicago.org/) and place it in `data/`.

## Workflow

Both pipelines are functionally equivalent:

| Stage | Ray | Spark |
|---|---|---|
| Ingestion | Size-based chunked loading | `read.csv()` with partition tuning |
| Preprocessing | Pandas-style cleaning, dedup, date features, label encoding | DataFrame ops, `StringIndexer`, `OneHotEncoder` |
| Feature engineering | PCA | `VectorAssembler`, `StandardScaler`, PCA (in a `Pipeline`) |
| Training / tuning | scikit-learn models + Ray Tune | MLlib + `CrossValidator` / `ParamGridBuilder` |
| Evaluation | Accuracy, AUC, F1 | `BinaryClassificationEvaluator` (+ accuracy, F1) |
| Inference test | Varying batch sizes | Varying batch sizes via `.limit()` |
| Monitoring | `RayPerformanceMonitor` (separate thread) | `SparkPerformanceMonitor` (psutil) |

**Models:** Random Forest, Decision Tree, Logistic Regression.

## Experimental setup

- **Platform:** Azure Machine Learning Studio, `Standard_F16s_v2` (16 vCPUs, 14 allocated to the frameworks; 32 GB RAM; 128 GB SSD)
- **OS / Python:** Ubuntu 20.04 LTS, Python 3.8
- **Ray:** 2.9.0 (`num_cpus=14`, `object_store_memory=10e9`), local cluster
- **Spark:** PySpark 3.4.1, local mode, dynamic allocation, 8 GB executor memory
- **Other libraries:** scikit-learn 1.3.0, Pandas 2.0.3, NumPy 1.24.3, Joblib 1.3.2, Matplotlib 3.7.2, Seaborn 0.12.2, psutil 5.9.5

## Key results

### Training time

| Framework | Model | 1M | 4M | Full |
|---|---|---|---|---|
| Ray | Random Forest | 93.9 s | 346.1 s | ~650 s |
| Ray | Decision Tree | 14.7 s | 49.0 s | ~90 s |
| Ray | Logistic Regression | 11.6 s | 16.4 s | ~40 s |
| Spark | Random Forest | 349.9 s | 544.0 s | 813.3 s |
| Spark | Decision Tree | 263.0 s | 355.7 s | 487.8 s |
| Spark | Logistic Regression | 306.0 s | 409.5 s | 566.0 s |

### Prediction throughput (batch size 1000 → 5000)

| Framework | Random Forest | Decision Tree | Logistic Regression |
|---|---|---|---|
| Ray | 27,983 → 189,306 rec/s | 2.5M → 7.8M rec/s | 2.8M → 16.9M rec/s |
| Spark | 85 → 363 rec/s | 81 → 400 rec/s | 79 → 386 rec/s |

### Summary of findings

- **Speed:** Ray was faster than Spark in training and far faster in prediction across all models and sizes.
- **Scaling:** Spark's training time grew more slowly with data size (1.8–2.3× from 1M to full) than Ray's (3.5–6.9×), although Ray stayed faster in absolute terms.
- **Memory:** Spark used 10–30% less memory on larger datasets.
- **CPU:** Spark reached ~100% peak CPU; Ray showed lower and more model-dependent usage.
- **Developer experience:** Ray needed roughly 25–30% fewer lines of code and felt more natural for Python-centric teams; Spark offers stronger enterprise integration and multi-language support.

### When to choose which

- **Ray:** time-critical applications, inference-heavy workloads, rapid ML experimentation, Python-first teams.
- **Spark:** large-scale enterprise deployments, memory-constrained environments, existing data-lake architectures.
- Hybrid architectures combining both are also viable.

## Limitations

- All experiments ran in **single-node / local mode** (one 16-vCPU VM), not on a multi-node cluster.
- Ray models are **scikit-learn** estimators parallelized with Ray, while Spark uses **MLlib**, so part of the gap reflects different library implementations and not only the platforms.
- In the Ray runs, Ray Tune trials used identical fixed hyperparameters, so no effective hyperparameter search was performed.
- Some model-quality results are weak (e.g. Ray Logistic Regression F1 = 0; Spark Decision Tree AUC = 0.42), so the comparison should be read mainly as a systems-performance benchmark, not a modelling one.
