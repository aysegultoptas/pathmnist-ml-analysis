from __future__ import annotations

import argparse
import os
import time
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==============================================================================
# GRAFİKLERİ ÇİZDİRMEK İSTİYOR MUSUNUZ? 
# GitHub için varsayılan olarak False (Kapalı). İsteyen True (Açık) yapabilir.
GRAFIKLERI_CIZ = False  
# ==============================================================================

RANDOM_STATE = 42

CLASS_NAMES = [
    'Adipose', 'Background', 'Debris', 'Lymphocytes',
    'Mucus', 'Smooth Muscle', 'Normal Colon Mucosa',
    'Cancer-Assoc. Stroma', 'Colorectal Adenocarcinoma'
]

SHORT_NAMES = [
    'Adipose', 'Backgr.', 'Debris', 'Lympho.',
    'Mucus', 'Sm.Musc', 'Norm.Col', 'CA.Str.', 'CRC'
]

ALL_CLASSIFIERS = ['SVM', 'RandomForest', 'LightGBM']
WEAK_CLASSES = ['Debris', 'Smooth Muscle', 'Cancer-Assoc. Stroma']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PathMNIST – Sınıflandırma")
    p.add_argument("--features-dir", type=str, default="outputs/features")
    p.add_argument("--outdir",       type=str, default="outputs/results")
    p.add_argument(
        "--clf",
        type=str,
        choices=ALL_CLASSIFIERS + ['all'],
        default='all',
        help="SVM | RandomForest | LightGBM | all"
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe(s: str) -> str:
    return (s.replace(' ', '_').replace('ü', 'u').replace('Ö', 'O').replace('ö', 'o').replace('.', ''))


def load_features(features_dir: Path):
    print(f"Öznitelikler yükleniyor: {features_dir}")
    train = np.load(features_dir / "X_train_scaled.npz")
    val   = np.load(features_dir / "X_val_scaled.npz")
    test  = np.load(features_dir / "X_test_scaled.npz")

    X_train_scaled = train['X'];  y_train = train['y']
    X_val_scaled   = val['X'];    y_val   = val['y']
    X_test_scaled  = test['X'];   y_test  = test['y']

    idx = np.load(features_dir / "feature_indices.npz")
    rgb_dim = int(idx['color_dim'][0]) if 'color_dim' in idx else (int(idx['rgb_dim'][0]) if 'rgb_dim' in idx else 96)

    feature_indices = {
        'all_features': idx['all_features'],
        'pearson':      idx['pearson'],
        'cfs':          idx['cfs'],
        'lasso':        idx['lasso'],
        'pso':          idx['pso'],
        'hog_dim':      int(idx['hog_dim'][0]),
        'lbp_dim':      int(idx['lbp_dim'][0]) if 'lbp_dim' in idx else 256,
        'rgb_dim':      rgb_dim,
        'gabor_dim':    int(idx['gabor_dim'][0]) if 'gabor_dim' in idx else 64,
    }
    return (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_indices)


def get_classifier(clf_name: str):
    if clf_name == 'SVM':
        return SVC(kernel='rbf', C=100, gamma='scale', random_state=RANDOM_STATE, cache_size=2000, class_weight='balanced')
    elif clf_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    elif clf_name == 'LightGBM':
        return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=127, min_child_samples=10, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight='balanced')
    raise ValueError(f"Bilinmeyen sınıflandırıcı: {clf_name}")


def train_evaluate(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    try:
        t0          = time.time()
        clf.fit(X_train, y_train)
        y_pred_val  = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        val_acc     = accuracy_score(y_val,  y_pred_val)
        test_acc    = accuracy_score(y_test, y_pred_test)
        elapsed     = time.time() - t0
        return test_acc, val_acc, y_pred_test, elapsed
    except Exception as e:
        print(f"  Model hata verdi: {e}")
        return 0.0, 0.0, None, 0.0


def save_confusion_matrix(y_test, y_pred, fs_name, clf_name, outdir: Path) -> None:
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    im0 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_xticks(range(9)); axes[0].set_yticks(range(9))
    axes[0].set_xticklabels(SHORT_NAMES, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(SHORT_NAMES, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(9):
        for j in range(9):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7, color='white' if cm[i, j] > thresh else 'black')
    axes[0].set_title('Confusion Matrix (Sayı)')

    im1 = axes[1].imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_xticks(range(9)); axes[1].set_yticks(range(9))
    axes[1].set_xticklabels(SHORT_NAMES, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(SHORT_NAMES, fontsize=8)
    for i in range(9):
        for j in range(9):
            axes[1].text(j, i, f'{cm_pct[i,j]:.1f}%', ha='center', va='center', fontsize=6, color='white' if cm_pct[i, j] > 50 else 'black')
    axes[1].set_title('Confusion Matrix (%)')
    fig.suptitle(f'{fs_name} + {clf_name}', fontsize=11)
    plt.tight_layout()
    fig.savefig(outdir / f"cm_{_safe(fs_name)}_{_safe(clf_name)}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_per_class_report(y_test, y_pred, fs_name, clf_name, outdir: Path) -> pd.DataFrame:
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    df.to_csv(outdir / f"report_{_safe(fs_name)}_{_safe(clf_name)}.csv")
    return df


def _feature_label(idx: int, hog_dim: int, lbp_dim: int, rgb_dim: int, gabor_dim: int) -> str:
    if idx < hog_dim: return f'HOG_{idx}'
    idx -= hog_dim
    if idx < lbp_dim: return f'LBP_{idx}'
    idx -= lbp_dim
    if idx < rgb_dim: return f'Color_{["R", "G", "B"][idx // (rgb_dim // 3)]}_{idx % (rgb_dim // 3)}'
    idx -= rgb_dim
    return f'Gabor_{idx}'


def plot_rf_feature_importance(clf, feature_indices, hog_dim, lbp_dim, rgb_dim, gabor_dim, outdir: Path, fs_name: str) -> None:
    if not hasattr(clf, 'feature_importances_'): return
    importances = clf.feature_importances_
    n           = len(importances)
    labels = [_feature_label(int(feature_indices[i]), hog_dim, lbp_dim, rgb_dim, gabor_dim) for i in range(n)]

    block_importance = {'HOG': 0.0, 'LBP': 0.0, 'Color': 0.0, 'Gabor': 0.0}
    for i, lbl in enumerate(labels):
        block = lbl.split('_')[0]
        if block in block_importance: block_importance[block] += importances[i]

    top_k   = min(30, n)
    top_idx = np.argsort(importances)[::-1][:top_k]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    block_colors = {'HOG': '#4C72B0', 'LBP': '#DD8452', 'Color': '#55A868', 'Gabor': '#C44E52'}
    bar_colors   = [block_colors.get(labels[i].split('_')[0], 'gray') for i in top_idx]
    axes[0].bar(range(top_k), importances[top_idx], color=bar_colors, alpha=0.85)
    axes[0].set_xticks(range(top_k))
    axes[0].set_xticklabels([labels[i] for i in top_idx], rotation=45, ha='right', fontsize=7)
    axes[0].set_title(f'Top {top_k} Öznitelik Önemi ({fs_name})')
    
    bl = {k: v for k, v in block_importance.items() if v > 0}
    axes[1].pie(bl.values(), labels=bl.keys(), autopct='%1.1f%%', colors=[block_colors[k] for k in bl])
    plt.tight_layout()
    fig.savefig(outdir / f'rf_importance_{_safe(fs_name)}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_f1(all_f1: dict, outdir: Path) -> None:
    n_combos = len(all_f1)
    fig, axes = plt.subplots(n_combos, 1, figsize=(12, 3.5 * n_combos), squeeze=False)
    for idx, ((fs_name, clf_name), f1_scores) in enumerate(all_f1.items()):
        ax     = axes[idx][0]
        colors = ['#d9534f' if f < 0.6 else '#5cb85c' if f >= 0.8 else '#f0ad4e' for f in f1_scores]
        bars   = ax.bar(CLASS_NAMES, f1_scores, color=colors, alpha=0.85, edgecolor='white')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{fs_name} + {clf_name}', fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=35, labelsize=7)
    plt.tight_layout()
    fig.savefig(outdir / 'per_class_f1_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_weak_class_comparison(all_f1: dict, outdir: Path) -> None:
    weak_indices = [CLASS_NAMES.index(c) for c in WEAK_CLASSES]
    combos       = list(all_f1.keys())
    if not combos: return
    fig, axes = plt.subplots(1, len(WEAK_CLASSES), figsize=(16, 5))
    for ax_i, (cls_name, cls_idx) in enumerate(zip(WEAK_CLASSES, weak_indices)):
        f1_vals = [all_f1[c][cls_idx] for c in combos]
        labels  = [f"{c[0][:10]}\n{c[1][:6]}" for c in combos]
        colors  = ['#d9534f' if v < 0.6 else '#5cb85c' if v >= 0.8 else '#f0ad4e' for v in f1_vals]
        axes[ax_i].bar(range(len(combos)), f1_vals, color=colors, alpha=0.85)
        axes[ax_i].set_xticks(range(len(combos)))
        axes[ax_i].set_xticklabels(labels, fontsize=6)
        axes[ax_i].set_ylim(0, 1.1)
        axes[ax_i].set_title(cls_name, fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / 'weak_class_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_precision_recall_f1_heatmap(all_reports: dict, outdir: Path) -> None:
    if not all_reports: return
    best_combo = max(all_reports, key=lambda k: all_reports[k]['test_acc'])
    report_df  = all_reports[best_combo]['report']
    data = [[report_df.loc[cls, 'precision'], report_df.loc[cls, 'recall'], report_df.loc[cls, 'f1-score']] if cls in report_df.index else [0,0,0] for cls in CLASS_NAMES]
    df_heat = pd.DataFrame(data, index=SHORT_NAMES, columns=['Precision', 'Recall', 'F1'])
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(df_heat, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, linewidths=0.5, ax=ax)
    plt.tight_layout()
    fig.savefig(outdir / 'best_combo_metrics_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_classifier_comparison_radar(all_f1: dict, outdir: Path) -> None:
    fs_names = ["Tüm Öznitelikler", "Pearson", "CFS", "LASSO", "PSO"]
    clf_names_plot = ['SVM', 'RandomForest', 'LightGBM']
    angles = np.linspace(0, 2 * np.pi, len(CLASS_NAMES), endpoint=False).tolist() + [0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()
    plot_idx = 0
    for fs_name in fs_names:
        if plot_idx >= len(axes_flat): break
        ax = axes_flat[plot_idx]
        for clf_name, color in zip(clf_names_plot, ['#4C72B0', '#DD8452', '#55A868']):
            key = (fs_name, clf_name)
            if key not in all_f1: continue
            values = all_f1[key] + [all_f1[key][0]]
            ax.plot(angles, values, color=color, linewidth=1.5, label=clf_name)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(SHORT_NAMES, fontsize=7); ax.set_ylim(0, 1)
        ax.set_title(fs_name, fontsize=9, fontweight='bold')
        plot_idx += 1
    if plot_idx < len(axes_flat): axes_flat[plot_idx].set_visible(False)
    plt.tight_layout()
    fig.savefig(outdir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_set_heatmap(all_f1: dict, outdir: Path) -> None:
    fs_names  = ["Tüm Öznitelikler", "Pearson", "CFS", "LASSO", "PSO"]
    clf_names = ['SVM', 'RandomForest', 'LightGBM']
    data = [[np.mean([all_f1[(fs, clf)][cls_idx] for clf in clf_names if (fs, clf) in all_f1]) for cls_idx in range(len(CLASS_NAMES))] for fs in fs_names]
    df_heat = pd.DataFrame(data, index=fs_names, columns=SHORT_NAMES)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df_heat, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0.3, vmax=1.0, linewidths=0.5, ax=ax)
    plt.tight_layout()
    fig.savefig(outdir / 'feature_set_class_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_overall_comparison(df_results: pd.DataFrame, outdir: Path) -> None:
    fs_order  = ["Tüm Öznitelikler", "Pearson", "CFS", "LASSO", "PSO"]
    clf_order = ["SVM", "RandomForest", "LightGBM"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_i, (metric, title) in enumerate([("Test Doğruluğu", "Test Accuracy (%)"), ("Val Doğruluğu",  "Validation Accuracy (%)")]):
        pivot = df_results.pivot(index="Öznitelik Seti", columns="Sınıflandırıcı", values=metric).reindex(fs_order).reindex(columns=[c for c in clf_order if c in df_results["Sınıflandırıcı"].unique()])
        x      = np.arange(len([f for f in fs_order if f in pivot.index]))
        width  = 0.25
        fs_avail = [f for f in fs_order if f in pivot.index]
        for j, (cn, color) in enumerate(zip(pivot.columns, ["#4C72B0", "#DD8452", "#55A868"])):
            vals = pivot.reindex(fs_avail)[cn].values * 100
            axes[ax_i].bar(x + j * width, vals, width, label=cn, color=color, alpha=0.85)
        axes[ax_i].set_xticks(x + width); axes[ax_i].set_xticklabels(fs_avail, rotation=15, ha='right', fontsize=9); ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(outdir / 'results_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_timing_analysis(df_timing: pd.DataFrame, outdir: Path) -> None:
    df_sorted = df_timing.sort_values('Süre (s)', ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.4)))
    ax.barh(df_sorted['Kombinasyon'], df_sorted['Süre (s)'], color='#55A868', alpha=0.85)
    plt.tight_layout()
    fig.savefig(outdir / 'timing_chart.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    global RANDOM_STATE
    RANDOM_STATE = int(args.seed); np.random.seed(RANDOM_STATE)

    features_dir = Path(args.features_dir)
    outdir       = ensure_dir(args.outdir)
    cm_dir       = ensure_dir(outdir / "confusion_matrices")
    model_dir    = ensure_dir(outdir / "models")

    clf_names = ALL_CLASSIFIERS if args.clf == 'all' else [args.clf]
    (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_indices) = load_features(features_dir)

    hog_dim = feature_indices['hog_dim']; lbp_dim = feature_indices['lbp_dim']
    rgb_dim = feature_indices['rgb_dim']; gabor_dim = feature_indices['gabor_dim']

    fs_map = {"Tüm Öznitelikler": feature_indices['all_features'], "Pearson": feature_indices['pearson'], "CFS": feature_indices['cfs'], "LASSO": feature_indices['lasso'], "PSO": feature_indices['pso']}
    feature_sets = {name: (X_train_scaled[:, idx], X_val_scaled[:, idx], X_test_scaled[:, idx], idx) for name, idx in fs_map.items()}

    results = {}; detailed_results = []; timing_records = []; all_f1 = {}; all_reports = {}

    for fs_name, (Xtr, Xvl, Xte, feat_idx) in feature_sets.items():
        print(f"\n■ Öznitelik Seti: {fs_name} ({Xtr.shape[1]} öznitelik)")
        for clf_name in clf_names:
            clf = get_classifier(clf_name)
            test_acc, val_acc, y_pred, elapsed = train_evaluate(clf, Xtr, y_train, Xvl, y_val, Xte, y_test)
            results[(fs_name, clf_name)] = test_acc

            detailed_results.append({"Öznitelik Seti": fs_name, "Sınıflandırıcı": clf_name, "Val Doğruluğu": val_acc, "Test Doğruluğu": test_acc, "Süre (s)": elapsed, "Feature_Count": Xtr.shape[1]})
            timing_records.append({"Kombinasyon": f"{fs_name} + {clf_name}", "Süre (s)": elapsed, "Feature_Count": Xtr.shape[1]})
            print(f"  {clf_name:20s} -> Val: {val_acc:.4f}  Test: {test_acc:.4f}  ({elapsed:.1f}s)")

            if y_pred is not None:
                if GRAFIKLERI_CIZ: save_confusion_matrix(y_test, y_pred, fs_name, clf_name, cm_dir)
                report_df = save_per_class_report(y_test, y_pred, fs_name, clf_name, outdir)
                f1_scores = [report_df.loc[cn, 'f1-score'] if cn in report_df.index else 0.0 for cn in CLASS_NAMES]
                all_f1[(fs_name, clf_name)]     = f1_scores
                all_reports[(fs_name, clf_name)] = {'report': report_df, 'test_acc': test_acc, 'val_acc': val_acc}
                if GRAFIKLERI_CIZ and clf_name == 'RandomForest':
                    plot_rf_feature_importance(clf, feat_idx, hog_dim, lbp_dim, rgb_dim, gabor_dim, outdir, fs_name)

            with open(model_dir / f"model_{_safe(fs_name)}_{_safe(clf_name)}.pkl", "wb") as f:
                pickle.dump(clf, f)

    df_results = pd.DataFrame(detailed_results); df_timing = pd.DataFrame(timing_records)
    df_results.to_csv(outdir / "results.csv", index=False); df_timing.to_csv(outdir / "timing.csv", index=False)

    if len(df_results) > 0:
        pivot_test = df_results.pivot(index="Öznitelik Seti", columns="Sınıflandırıcı", values="Test Doğruluğu").reindex(fs_order)
        print("\n[BAŞARILI] Sınıflandırma raporları 'outputs/results/results.csv' dosyasına kaydedildi.")
        print("\nTest Doğruluğu Sonuç Özeti (%):\n")
        print((pivot_test * 100).round(2).to_string())

    if GRAFIKLERI_CIZ:
        plot_overall_comparison(df_results, outdir)
        if all_f1:
            plot_per_class_f1(all_f1, outdir); plot_weak_class_comparison(all_f1, outdir); plot_feature_set_heatmap(all_f1, outdir); plot_classifier_comparison_radar(all_f1, outdir)
        if all_reports: plot_precision_recall_f1_heatmap(all_reports, outdir)
        if len(df_timing) > 0: plot_timing_analysis(df_timing, outdir)


if __name__ == "__main__":
    main()
