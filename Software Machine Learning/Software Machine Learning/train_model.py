"""
Boxing Punch Classification — ML Training Pipeline
====================================================
Trains multiple classifiers on extracted IMU features
and evaluates performance for thesis.

Usage:
    python train_model.py

Output:
    - ml_data/model_results.txt     : accuracy scores and classification report
    - ml_data/confusion_matrix.png  : confusion matrix plots
    - ml_data/feature_importance.png: top features plot
    - ml_data/best_model.pkl        : saved best model for future use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (LeaveOneOut, cross_val_score,
                                     StratifiedKFold, cross_val_predict)
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
ML_DIR     = os.path.join(DATA_DIR, 'ml_data')
FEATURES_PATH = os.path.join(ML_DIR, 'features_all.csv')

LABEL_COL  = 'label'
DROP_COLS  = ['label', 'session', 'device_id', 'location',
              'peak_time_ms', 'n_samples']

# Labels to include (skip Rest/Unknown if present)
VALID_LABELS = ['Jab', 'Cross', 'Hook', 'Uppercut',
                'Roundhouse_Right', 'Roundhouse_Left',
                'LowKick_Right', 'LowKick_Left']

# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        print("Run preprocess.py first!")
        exit(1)

    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} windows with {len(df.columns)} columns")

    # Filter to valid labels only
    df = df[df[LABEL_COL].isin(VALID_LABELS)].copy()
    print(f"After filtering labels: {len(df)} windows")
    print(f"Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")

    return df

# ── Prepare Features ──────────────────────────────────────────────────────────

def prepare_features(df):
    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df[LABEL_COL].copy()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Classes: {list(le.classes_)}")

    return X, y_encoded, le, feature_cols

# ── Models ────────────────────────────────────────────────────────────────────

def get_models():
    return {
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
            ('clf',     RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ]),
        'SVM (RBF)': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
            ('clf',     SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            ))
        ]),
        'Gradient Boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
            ('clf',     GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            ))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
            ('clf',     KNeighborsClassifier(
                n_neighbors=3,
                weights='distance'
            ))
        ]),
    }

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_models(X, y, le, n_samples):
    """
    Choose validation strategy based on dataset size.
    With small datasets: Leave-One-Out cross validation
    With larger datasets: Stratified K-Fold
    """
    models = get_models()
    results = {}

    # Choose CV strategy
    if n_samples < 50:
        cv = LeaveOneOut()
        cv_name = "Leave-One-Out"
        print(f"\nUsing {cv_name} CV (small dataset: {n_samples} samples)")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "5-Fold Stratified"
        print(f"\nUsing {cv_name} CV ({n_samples} samples)")

    print(f"\n{'='*60}")
    print("Model Evaluation Results")
    print(f"{'='*60}")

    best_model = None
    best_score = 0
    best_name  = ""

    for name, model in models.items():
        print(f"\nTraining: {name}...")
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            mean_acc = scores.mean()
            std_acc  = scores.std()

            # Get predictions for confusion matrix
            y_pred = cross_val_predict(model, X, y, cv=cv)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

            results[name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy':  std_acc,
                'f1_weighted':   f1,
                'y_pred':        y_pred,
                'scores':        scores,
            }

            print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"  F1 (weighted): {f1:.3f}")

            if mean_acc > best_score:
                best_score = mean_acc
                best_name  = name
                best_model = model

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {'mean_accuracy': 0, 'std_accuracy': 0,
                            'f1_weighted': 0, 'y_pred': y, 'scores': [0]}

    print(f"\n{'='*60}")
    print(f"Best model: {best_name} (accuracy={best_score:.3f})")
    print(f"{'='*60}")

    return results, best_model, best_name, cv_name

# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results, y, le):
    """Plot confusion matrix for each model."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    class_names = le.classes_

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y, res['y_pred'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"{name}\nAcc={res['mean_accuracy']:.2f} F1={res['f1_weighted']:.2f}",
                    fontsize=9)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm[i, j]
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                       color=color, fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=axes[-1], fraction=0.046)
    plt.suptitle('Confusion Matrices — Boxing Punch Classification', fontsize=12, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(ML_DIR, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {out_path}")

def plot_feature_importance(best_model, feature_cols, best_name):
    """Plot top 20 most important features from Random Forest."""
    try:
        # Get the classifier from pipeline
        clf = best_model.named_steps['clf']
        if not hasattr(clf, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:20]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(indices)),
               importances[indices],
               color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_cols[i] for i in indices],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Feature Importance')
        ax.set_title(f'Top 20 Features — {best_name}')
        plt.tight_layout()

        out_path = os.path.join(ML_DIR, 'feature_importance.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature importance saved: {out_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

def plot_accuracy_comparison(results):
    """Bar chart comparing model accuracies."""
    names  = list(results.keys())
    means  = [r['mean_accuracy'] for r in results.values()]
    stds   = [r['std_accuracy']  for r in results.values()]
    f1s    = [r['f1_weighted']   for r in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, means, width, yerr=stds,
                   label='Accuracy', color='steelblue', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, f1s, width,
                   label='F1 (weighted)', color='coral', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Boxing Punch Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=1,
               label='Random baseline (4 classes)')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(ML_DIR, 'model_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved: {out_path}")

# ── Save Results ──────────────────────────────────────────────────────────────

def save_results(results, y, le, cv_name, best_name, n_samples, n_features):
    lines = []
    lines.append("Boxing Punch Classification — ML Results")
    lines.append("="*60)
    lines.append(f"Dataset: {n_samples} windows, {n_features} features")
    lines.append(f"Classes: {list(le.classes_)}")
    lines.append(f"Validation: {cv_name}")
    lines.append(f"Best model: {best_name}")
    lines.append("")

    for name, res in results.items():
        lines.append(f"\n--- {name} ---")
        lines.append(f"Accuracy: {res['mean_accuracy']:.3f} ± {res['std_accuracy']:.3f}")
        lines.append(f"F1 (weighted): {res['f1_weighted']:.3f}")
        lines.append("\nClassification Report:")
        report = classification_report(y, res['y_pred'],
                                       target_names=le.classes_,
                                       zero_division=0)
        lines.append(report)

    out_path = os.path.join(ML_DIR, 'model_results.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Results saved: {out_path}")

def save_best_model(best_model, X, y, le, feature_cols):
    """Retrain best model on all data and save."""
    best_model.fit(X, y)
    model_data = {
        'model':        best_model,
        'label_encoder': le,
        'feature_cols': feature_cols,
    }
    out_path = os.path.join(ML_DIR, 'best_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Best model saved: {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\nBoxing Punch Classification — ML Training")
    print("="*60)

    # Load and prepare
    df = load_data()

    if len(df) < 4:
        print("\nERROR: Too few samples to train. Need at least 4 windows.")
        print("Re-run label_boxing_data.py with lower threshold and preprocess.py")
        return

    X, y, le, feature_cols = prepare_features(df)

    # Convert to numpy
    X_vals = X.values.astype(float)

    # Warn if dataset is very small
    if len(df) < 30:
        print(f"\n⚠ WARNING: Only {len(df)} windows available.")
        print("Results may not be reliable — collect more data for robust ML.")
        print("This is suitable for proof-of-concept demonstration.\n")

    # Train and evaluate
    results, best_model, best_name, cv_name = evaluate_models(
        X_vals, y, le, len(df)
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_confusion_matrices(results, y, le)
    plot_accuracy_comparison(results)
    if best_name == 'Random Forest' or best_name == 'Gradient Boosting':
        plot_feature_importance(best_model, feature_cols, best_name)

    # Save results and model
    save_results(results, y, le, cv_name, best_name, len(df), X_vals.shape[1])
    save_best_model(best_model, X_vals, y, le, feature_cols)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"All outputs saved to: {ML_DIR}")
    print("\nFiles generated:")
    print("  model_results.txt     — accuracy and classification report")
    print("  confusion_matrix.png  — confusion matrices for all models")
    print("  model_comparison.png  — accuracy comparison bar chart")
    print("  feature_importance.png— top 20 most important features")
    print("  best_model.pkl        — saved model for deployment")

if __name__ == '__main__':
    main()
