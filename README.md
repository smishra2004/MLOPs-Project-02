# 🫁 Chest X-Ray Pneumonia Classifier — End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![AWS](https://img.shields.io/badge/AWS-S3-orange?logo=amazonaws&logoColor=white)](https://aws.amazon.com/s3/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow&logoColor=white)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Model%20Registry-0969DA?logo=dagshub&logoColor=white)](https://dagshub.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 📌 Project Overview

A **production-grade, end-to-end deep learning pipeline** that classifies chest X-ray images as **NORMAL** or **PNEUMONIA**, built with a full MLOps lifecycle — from cloud data ingestion through governed model deployment and live inference. This project demonstrates the ability to architect, build, and operate scalable **computer vision ML systems** that mirror real-world industry workflows.

> **Business Value:** Enables healthcare teams to accelerate preliminary X-ray screening, reduce radiologist workload on high-volume cases, and deploy continuously improving models with full experiment traceability — cutting triage time while maintaining governed, auditable model promotion.

---

## 🏗️ System Architecture

```
AWS S3 (Dataset + Model Registry)
        │
        ▼
┌───────────────────┐
│   Data Ingestion  │ ──► Downloads zip from S3, extracts train/test/val splits
└───────────────────┘
        │
        ▼
┌──────────────────────┐
│ Data Transformation  │ ──► Augmentations, ImageFolder datasets, weighted sampling
└──────────────────────┘
        │
        ▼
┌───────────────────┐
│   Model Trainer   │ ──► ResNet50 fine-tuning with MLflow experiment tracking
└───────────────────┘
        │
        ▼
┌──────────────────────┐
│  Model Evaluation   │ ──► ROC-AUC comparison vs. production model in S3
└──────────────────────┘
        │
        ▼
┌───────────────────┐
│   Model Pusher    │ ──► Promotes champion model to S3 (only if AUC improves)
└───────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Streamlit App (app.py)      │ ──► Upload X-ray → predict · Sidebar → retrain
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  DagsHub + MLflow — params, metrics, artifacts   │
└──────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch, torchvision (ResNet50) |
| **Data Store** | AWS S3 (dataset zip + model registry) |
| **Experiment Tracking** | MLflow + DagsHub |
| **Web UI** | Streamlit |
| **Metrics** | scikit-learn (AUC, precision, recall, F1) |
| **Cloud SDK** | boto3, mypy-boto3-s3 |
| **Package Management** | pip, `setup.py` + `pyproject.toml` |
| **Cloud IAM** | AWS IAM (least-privilege access keys) |
| **Config Management** | Dataclasses, environment variables (`.env`) |

---

## 🚀 Key Features & Engineering Decisions

### ✅ Modular Pipeline Architecture
Each pipeline stage (ingestion → transformation → training → evaluation → pushing) is an independent, reusable component with dedicated `Config` and `Artifact` entity classes — enabling isolated testing, easy debugging, and clean handoffs between stages.

### ✅ Cloud-Native Data & Model Layer
The full dataset (`chest-deep-learning.zip`) and production model live in **AWS S3**, pulled programmatically at runtime. This eliminates static file dependencies, mirrors real enterprise ML architectures, and keeps training reproducible from a single remote source of truth.

### ✅ Automated Model Governance
The **Model Evaluation** component compares newly trained models against the current production model in S3. A model only gets promoted if its **ROC-AUC beats the production baseline** — preventing regressions from reaching production.

### ✅ PyTorch-Aware S3 Model Registry
`ChestXrayEstimator` bypasses pickle-based loading and uses `torch.load()` for `.pth` checkpoints — a production detail that correctly handles PyTorch weights where naive sklearn-style registries fail.

### ✅ EDA-Driven ML Engineering
Notebooks in `/experiments` directly informed design: **199 unique image sizes** → 224×224 resize; **~2.89× class imbalance** → weighted loss + `WeightedRandomSampler`; medically realistic augmentations only.

### ✅ Full Experiment Observability
Every training run logs hyperparameters, per-epoch metrics, test-set metrics, confusion matrix PNG, and model weights to **MLflow via DagsHub** — giving complete audit trails for every experiment.

### ✅ Clean Package Structure
The project is installable as a local Python package via `setup.py` and `pyproject.toml`, enabling clean imports across modules and following software engineering best practices beyond notebook-style code.

### ✅ Structured Logging & Custom Exception Handling
All pipeline stages use a centralized logger and custom `MyException` class — making debugging across distributed components significantly faster.

---

## 📂 Project Structure

```
MLOPs-Project-02/
│
├── src/
│   ├── components/               # Pipeline stage implementations
│   │   ├── data_ingestion.py
│   │   ├── chest_xray_transforms.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/            # Service connection configs
│   │   ├── aws_connection.py
│   │   └── aws_dataset_upload.py
│   │
│   ├── entity/                   # Config & Artifact dataclasses
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   └── chest_xray_s3_estimator.py
│   │
│   ├── pipeline/                 # Training & prediction orchestration
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── data_access/              # S3 dataset download & extraction
│   ├── cloud_storage/            # S3 push/pull utilities
│   ├── constants/                # Global constants (bucket, keys, paths)
│   ├── logger/                   # Rotating file + console logging
│   └── exception/                # Custom exception handling
│
├── config/
│   ├── model.yaml                # Hyperparameter config (placeholder)
│   └── schema.yaml               # Dataset schema (placeholder)
│
├── experiments/                  # EDA notebooks & validation plots
├── app.py                        # Streamlit app — prediction + retrain trigger
├── demo.py                       # Local pipeline test runner (CLI)
├── Dockerfile                    # Container scaffold (placeholder)
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## ⚙️ Local Setup

### 1. Clone & Create Environment
```bash
git clone https://github.com/smishra2004/MLOPs-Project-02.git
cd MLOPs-Project-02

conda create -n chest-xray python=3.10 -y
conda activate chest-xray
pip install -r requirements.txt
pip install -e .
pip install scikit-learn Pillow
```

### 2. Set Environment Variables
Create a `.env` file in the project root:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID="<your-access-key>"
AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

For MLflow/DagsHub tracking, authenticate via [DagsHub CLI](https://dagshub.com/docs/integration_guide/mlflow/) or set your token as an environment variable.

### 3. Run Training Pipeline
```bash
python demo.py
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
# Visit: http://localhost:8501
```

Upload a chest X-ray (JPG/PNG) for instant prediction, or click **🚀 Train Model** in the sidebar to trigger the full retraining pipeline.

---

## ☁️ Cloud Infrastructure Setup

### AWS S3
| Resource | Purpose |
|---|---|
| **Bucket** (`chest-x-ray-dataset-cnn`) | Stores dataset zip and production model |
| **Key** `chest-deep-learning.zip` | Full chest X-ray dataset (train/test/val splits) |
| **Key** `cnn_model.pkl` | Production ResNet50 weights (PyTorch `.pth` format) |
| **Region** | `us-east-1` |

### AWS IAM
1. Create an IAM user with scoped S3 permissions (read/write on the bucket)
2. Generate an access key pair
3. Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`

### DagsHub + MLflow
| Resource | Purpose |
|---|---|
| **Repo** | [smishra2004/MLOPs-Project-02](https://dagshub.com/smishra2004/MLOPs-Project-02) |
| **MLflow URI** | `https://dagshub.com/smishra2004/MLOPs-Project-02.mlflow` |
| **Experiment** | `chest-xray-detection-experiment` |

Every pipeline run logs params, per-epoch metrics, test metrics, confusion matrix, and model artifact automatically.

---

## 🖥️ Streamlit App Features

| Feature | Description |
|---|---|
| **X-Ray Upload** | Drag-and-drop JPG/PNG chest X-ray for instant classification |
| **Live Prediction** | Returns **NORMAL** or **PNEUMONIA** with visual result styling |
| **On-Demand Retraining** | Sidebar **Train Model** button triggers the full MLOps pipeline |
| **S3 Model Loading** | Production model loaded lazily from S3 on first prediction |
| **Medical Disclaimer** | Built-in notice that AI output is not a substitute for clinical diagnosis |

---

## 🧠 ML Pipeline Details

- **Model:** ResNet50 (ImageNet pretrained) with frozen backbone + custom 2-class head
- **Classes:** `NORMAL (0)` · `PNEUMONIA (1)`
- **Loss:** Weighted `CrossEntropyLoss` + `WeightedRandomSampler` for ~2.89× class imbalance
- **Optimizer:** Adam (lr=1e-4) with `ReduceLROnPlateau` scheduler
- **Early stopping:** Patience = 3 on validation loss
- **Primary metric:** ROC-AUC on held-out test set
- **Promotion rule:** New model accepted if `trained_AUC > production_AUC` (first deployment auto-accepted)
- **Augmentations (train only):** Horizontal flip, ±10° rotation, brightness jitter, subtle crop — no medically unrealistic transforms
- **EDA & validation** documented in `/experiments` notebooks

### Model Promotion Logic
```
IF no model in S3          → accept new model (first deployment)
IF trained_AUC > prod_AUC  → push to S3, replace production
ELSE                       → reject, keep existing production model
```

---

## 🌱 What This Project Demonstrates

| Skill Area | Demonstrated By |
|---|---|
| **MLOps Engineering** | Full pipeline with automated retraining, AUC-based evaluation gates, and S3 model registry |
| **Deep Learning** | Transfer learning with ResNet50, class imbalance handling, medically informed augmentations |
| **Cloud Architecture** | AWS S3 for dataset storage and model versioning with IAM-scoped credentials |
| **Experiment Tracking** | MLflow + DagsHub integration with params, metrics, and artifact logging |
| **Software Engineering** | Modular OOP design, typed artifact contracts, custom packaging, centralized logging |
| **Computer Vision** | End-to-end image classification pipeline from raw X-rays to production inference |
| **Production Thinking** | Threshold-based model governance, lazy S3 model loading, environment variable management |

---

## 🗺️ Roadmap

- [ ] Containerize with Docker (`Dockerfile` scaffold exists)
- [ ] Add GitHub Actions CI/CD (lint, test, deploy)
- [ ] Implement data validation stage
- [ ] Externalize hyperparameters via `config/model.yaml`
- [ ] Add unit & integration tests
- [ ] Display prediction confidence scores in Streamlit UI

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes**. AI-assisted predictions are not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## 👤 Contact

**Shubham Mishra**  
📧 shubham.smishra2004@gmail.com  
🔗 [GitHub](https://github.com/smishra2004) · [DagsHub](https://dagshub.com/smishra2004/MLOPs-Project-02)

---

> Built to production standards — not just to run in a notebook.
