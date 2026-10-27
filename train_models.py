"""
train_models.py - Model Training and Evaluation Pipeline

This script performs reproducible training, grouped cross-validation, holdout evaluation,
dynamic model selection, and artifact serialization according to the project specification.
"""

import json
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Random Seed for reproducibility
RANDOM_SEED = 42

# Permitted 9 model features and target
NUMERIC_FEATURES = [
    "witnesses",
    "total_claim_amount",
    "policy_deductable",
    "months_as_customer",
    "number_of_vehicles_involved",
    "incident_hour_of_the_day",
]

CATEGORICAL_FEATURES = [
    "incident_severity",
    "police_report_available",
    "authorities_contacted",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "claim_status"
GROUP_COLUMN = "source_profile_id"
ALLOWED_LABELS = ["Approved", "Rejected"]


def resolve_data_path() -> str:
    """Find the path to the cleaned CSV file."""
    candidates = [
        os.path.join("data", "insurance_claim_dataset_cleaned.csv"),
        "insurance_claim_dataset_cleaned.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cleaned dataset not found in expected locations: {candidates}"
    )


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """Load and audit dataset according to the data contract."""
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from '{csv_path}' with shape: {df.shape}")

    # Verify target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET_COLUMN}'")
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset(set(ALLOWED_LABELS)):
        raise ValueError(f"Unexpected target labels found: {unique_targets}")

    # Verify grouping column
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Missing grouping column: '{GROUP_COLUMN}'")

    # Verify all 9 features are present
    missing_features = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model features in dataset: {missing_features}")

    # Verify no nulls in required columns
    null_counts = df[MODEL_FEATURES + [TARGET_COLUMN, GROUP_COLUMN]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values detected in core columns:\n{null_counts[null_counts > 0]}")

    print("Data contract audit passed successfully.")
    return df


def deterministic_grouped_split(df: pd.DataFrame, holdout_group_count: int = 197):
    """
    Split the dataset using deterministic grouped splitting on source_profile_id.
    Ensures zero group leakage between training and holdout partitions.
    """
    unique_groups = np.sort(df[GROUP_COLUMN].unique())
    total_groups = len(unique_groups)

    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_groups = rng.permutation(unique_groups)

    holdout_groups = set(shuffled_groups[:holdout_group_count])
    train_groups = set(shuffled_groups[holdout_group_count:])

    # Assert no overlap
    overlap = train_groups.intersection(holdout_groups)
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} groups overlap!"

    train_mask = df[GROUP_COLUMN].isin(train_groups)
    holdout_mask = df[GROUP_COLUMN].isin(holdout_groups)

    train_df = df[train_mask].copy()
    holdout_df = df[holdout_mask].copy()

    print(f"Grouped Split Summary:")
    print(f"  Total groups: {total_groups}")
    print(f"  Train groups: {len(train_groups)}, Train rows: {len(train_df)}")
    print(f"  Holdout groups: {len(holdout_groups)}, Holdout rows: {len(holdout_df)}")
    print(f"  Train class distribution:\n{train_df[TARGET_COLUMN].value_counts(normalize=True).to_dict()}")
    print(f"  Holdout class distribution:\n{holdout_df[TARGET_COLUMN].value_counts(normalize=True).to_dict()}")

    return train_df, holdout_df, train_groups, holdout_groups


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models():
    """Return dictionary of the four candidate classifiers."""
    from sklearn.tree import DecisionTreeClassifier

    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def plot_confusion_matrices(eval_results: dict, output_dir: str):
    """Save confusion matrix figures for all evaluated models."""
    os.makedirs(output_dir, exist_ok=True)
    for model_name, res in eval_results.items():
        cm = res["confusion_matrix"]
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"cm_{model_name.lower()}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()


def plot_comparison_chart(comparison_df: pd.DataFrame, output_dir: str):
    """Save model accuracy and macro F1 comparison chart."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=comparison_df, x="model", y="holdout_accuracy", hue="model", legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title("Holdout Accuracy Comparison")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Model")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=comparison_df, x="model", y="holdout_macro_f1", hue="model", legend=False, ax=axes[1], palette="magma")
    axes[1].set_title("Holdout Macro F1 Score Comparison")
    axes[1].set_ylim(0.7, 1.0)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xlabel("Model")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()


def train_and_evaluate():
    """Main execution function."""
    csv_path = resolve_data_path()
    df = load_and_validate_data(csv_path)

    train_df, holdout_df, train_groups, holdout_groups = deterministic_grouped_split(df)

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    groups_train = train_df[GROUP_COLUMN]

    X_holdout = holdout_df[MODEL_FEATURES]
    y_holdout = holdout_df[TARGET_COLUMN]

    candidate_models = get_candidate_models()
    gkf = GroupKFold(n_splits=5)

    eval_results = {}
    comparison_records = []
    fitted_pipelines = {}

    print("\n--- Training and Cross-Validation (5-Fold GroupKFold) ---")
    for model_name, classifier in candidate_models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        # 5-fold GroupKFold cross-validation on train split only
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, groups=groups_train, cv=gkf, scoring="accuracy"
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        # Fit on complete training partition
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        # Evaluate on untouched holdout split
        y_pred = pipeline.predict(X_holdout)

        # Approved class probability score
        approved_idx = list(pipeline.classes_).index("Approved")
        y_proba = pipeline.predict_proba(X_holdout)[:, approved_idx]

        holdout_acc = float(accuracy_score(y_holdout, y_pred))
        holdout_bal_acc = float(balanced_accuracy_score(y_holdout, y_pred))
        holdout_macro_f1 = float(f1_score(y_holdout, y_pred, average="macro"))
        roc_auc = float(roc_auc_score((y_holdout == "Approved").astype(int), y_proba))
        cm = confusion_matrix(y_holdout, y_pred, labels=["Rejected", "Approved"]).tolist()
        report = classification_report(y_holdout, y_pred, output_dict=True)

        eval_results[model_name] = {
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "holdout_accuracy": holdout_acc,
            "holdout_balanced_accuracy": holdout_bal_acc,
            "holdout_macro_f1": holdout_macro_f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        comparison_records.append({
            "model": model_name,
            "cv_accuracy_mean": round(cv_mean, 4),
            "cv_accuracy_std": round(cv_std, 4),
            "holdout_accuracy": round(holdout_acc, 4),
            "holdout_balanced_accuracy": round(holdout_bal_acc, 4),
            "holdout_macro_f1": round(holdout_macro_f1, 4),
            "roc_auc": round(roc_auc, 4),
        })

        print(f"[{model_name}] CV: {cv_mean:.4f} ± {cv_std:.4f} | Holdout Acc: {holdout_acc:.4f} | Macro F1: {holdout_macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    comparison_df = pd.DataFrame(comparison_records)

    # Dynamic model selection: model with highest holdout accuracy
    best_model_row = comparison_df.sort_values(by=["holdout_accuracy", "holdout_macro_f1"], ascending=False).iloc[0]
    best_model_name = best_model_row["model"]
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"\n==========================================")
    print(f"Selected Best Model: {best_model_name}")
    print(f"Selection Metric: Highest Holdout Accuracy ({best_model_row['holdout_accuracy']:.4f})")
    print(f"==========================================\n")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    pipeline_path = os.path.join("models", "claim_model_pipeline.joblib")
    joblib.dump(best_pipeline, pipeline_path)
    print(f"Saved complete pipeline to: {pipeline_path}")

    # Metadata
    metadata = {
        "selected_model": best_model_name,
        "selection_rationale": f"Selected dynamically based on highest holdout accuracy ({best_model_row['holdout_accuracy']}) and balanced F1 ({best_model_row['holdout_macro_f1']}) across 4 candidate models.",
        "random_seed": RANDOM_SEED,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "total_count": len(MODEL_FEATURES),
        },
        "split": {
            "method": "Deterministic grouped split on source_profile_id",
            "train_groups": len(train_groups),
            "holdout_groups": len(holdout_groups),
            "train_rows": len(train_df),
            "holdout_rows": len(holdout_df),
        },
        "performance": eval_results[best_model_name],
        "library_versions": {
            "python": sys.version,
            "scikit_learn": getattr(sys.modules.get("sklearn"), "__version__", "unknown"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")

    # Comparison report
    comparison_path = os.path.join("reports", "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "models_evaluated": list(candidate_models.keys()),
            "results": eval_results,
            "selected_model": best_model_name,
        }, f, indent=2)
    print(f"Saved model comparison to: {comparison_path}")

    # Generate figures
    plot_confusion_matrices(eval_results, "reports/figures")
    plot_comparison_chart(comparison_df, "reports/figures")
    print("Saved comparison plots to 'reports/figures/'.")

    # Reload test
    reloaded_pipeline = joblib.load(pipeline_path)
    sample_data = pd.DataFrame([{
        "incident_severity": "Major Damage",
        "police_report_available": "YES",
        "authorities_contacted": "Police",
        "witnesses": 2,
        "total_claim_amount": 45000,
        "policy_deductable": 1000,
        "months_as_customer": 120,
        "number_of_vehicles_involved": 2,
        "incident_hour_of_the_day": 14,
    }])
    sample_pred = reloaded_pipeline.predict(sample_data)[0]
    sample_score = reloaded_pipeline.predict_proba(sample_data)[0, list(reloaded_pipeline.classes_).index("Approved")]
    print(f"Smoke test reload prediction: {sample_pred} (approval score: {sample_score:.4f})")
    print("Training pipeline finished successfully.")


if __name__ == "__main__":
    train_and_evaluate()
