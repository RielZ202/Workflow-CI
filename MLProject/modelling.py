"""
modelling.py (MLProject)
========================
Script training model untuk MLflow Project & GitHub Actions CI.
Mendukung argumen CLI agar bisa dikonfigurasi melalui MLProject entry point.

Usage:
    python modelling.py [--n_estimators 200] [--max_depth 12] ...
"""

import os
import sys
import json
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train AI Student Burnout Risk Model')
parser.add_argument('--n_estimators',      type=int,   default=200)
parser.add_argument('--max_depth',         type=int,   default=12)
parser.add_argument('--min_samples_split', type=int,   default=2)
parser.add_argument('--max_features',      type=str,   default='sqrt')
parser.add_argument('--random_state',      type=int,   default=42)
args = parser.parse_args()


# ─────────────────────────────────────────────────────────
# KONFIGURASI MLFLOW
# ─────────────────────────────────────────────────────────
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '').strip()

if not TRACKING_URI:
    print("ERROR: MLFLOW_TRACKING_URI environment variable is not set!")
    print("Please set MLFLOW_TRACKING_URI before running this script.")
    sys.exit(1)

# Validate URI format
if not TRACKING_URI.startswith('http'):
    print(f"ERROR: Invalid MLFLOW_TRACKING_URI format: {TRACKING_URI}")
    print("URI must start with 'http://' or 'https://'")
    sys.exit(1)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("AI-Student-BurnoutRisk-CI")

print(f"MLflow Tracking URI: {TRACKING_URI}")

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'train_processed.csv')
TEST_PATH  = os.path.join(BASE_DIR, 'test_processed.csv')

df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

TARGET  = 'Burnout_Risk_Level'
CLASSES = ['Low', 'Medium', 'High']

X_train = df_train.drop(columns=[TARGET])
y_train = df_train[TARGET]
X_test  = df_test.drop(columns=[TARGET])
y_test  = df_test[TARGET]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────
# TRAINING DAN LOGGING
# ─────────────────────────────────────────────────────────
with mlflow.start_run(run_name="CI_RandomForest", nested=True):
    # Model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_split=args.min_samples_split,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    y_bin       = label_binarize(y_test, classes=[0, 1, 2])

    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, average='weighted')
    rec      = recall_score(y_test, y_pred, average='weighted')
    f1_w     = f1_score(y_test, y_pred, average='weighted')
    f1_mac   = f1_score(y_test, y_pred, average='macro')
    roc_auc  = roc_auc_score(y_bin, y_pred_prob, multi_class='ovr', average='weighted')

    # Log Params
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("max_features",      args.max_features)
    mlflow.log_param("random_state",      args.random_state)

    # Log Metrics
    mlflow.log_metric("accuracy",           acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted",    rec)
    mlflow.log_metric("f1_weighted",        f1_w)
    mlflow.log_metric("f1_macro",           f1_mac)
    mlflow.log_metric("roc_auc_weighted",   roc_auc)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="AIStudent-BurnoutRisk-CI"
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title('Confusion Matrix - CI Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, 'training_confusion_matrix.png')
    plt.savefig(cm_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Classification Report JSON
    report_dict = classification_report(
        y_test, y_pred, target_names=CLASSES, output_dict=True
    )
    report_path = os.path.join(BASE_DIR, 'metric_info.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    mlflow.log_artifact(report_path)

    run_id = mlflow.active_run().info.run_id
    print(f"\n=== Training Selesai! ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-Score  : {f1_w:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Run ID    : {run_id}")

    # Simpan run_id untuk digunakan oleh workflow berikutnya
    with open(os.path.join(BASE_DIR, 'latest_run_id.txt'), 'w') as f:
        f.write(run_id)
