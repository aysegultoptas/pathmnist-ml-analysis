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
# Gelişmiş Grafikleri Çizdirmek İstiyor musunuz? 
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

# Zayıf sınıflar 
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


def load_dataset(features_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    path = features_dir / f"X_{name}_scaled.npz"
    if not path.exists():
        raise FileNotFoundError(f"Öznitelik dosyası bulunamadı: {path}")
    data = np.load(path)
    X = data['X']
    y = data['y'].flatten()
    feat_idx = data['feat_idx'].tolist() if 'feat_idx' in data else None
    return X, y, feat_idx


def get_classifier(name: str, seed: int) -> SVC | RandomForestClassifier | lgb.LGBMClassifier:
    if name == 'SVM':
        return SVC(
            C=100,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            random_state=seed,
            cache_size=2000
        )
    elif name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=300,
            max_features='sqrt',
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
    elif name == 'LightGBM':
        return lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=10,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
            verbosity=-1
        )
    raise ValueError(f"Bilinmeyen sınıflandırıcı: {name}")


def train_evaluate(clf, X_train, y_train, X_val, y_val, X_test, y_test) -> tuple[float, float, np.ndarray | None, float]:
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    return test_acc, val_acc, test_preds, elapsed


def save_confusion_matrix(y_true, y_pred, fs_name: str, clf_name: str, cm_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = [
        f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    ]
    labels = np.asarray(labels).reshape(cm.shape[0], cm.shape[1])

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
    plt.title(f"Confusion Matrix: {fs_name} + {clf_name}")
    plt.ylabel("Gerçek Sınıf")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.tight_layout()

    filename = f"cm_{_safe(fs_name)}_{_safe(clf_name)}.png"
    plt.savefig(cm_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_per_class_report(y_true, y_pred, fs_name: str, clf_name: str, outdir: Path) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    df = pd.DataFrame(rep).transpose()
    filename = f"report_{_safe(fs_name)}_{_safe(clf_name)}.csv"
    df.to_csv(outdir / filename)
    return df


def plot_rf_feature_importance(clf, feat_idx: list[int] | None, h_dim: int, l_dim: int, c_dim: int, g_dim: int, outdir: Path, fs_name: str):
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_

    actual_indices = feat_idx if feat_idx is not None else list(range(len(importances)))

    h_idx = [i for i in actual_indices if i < h_dim]
    l_idx = [i for i in actual_indices if h_dim <= i < h_dim + l_dim]
    c_idx = [i for i in actual_indices if h_dim + l_dim <= i < h_dim + l_dim + c_dim]
    g_idx = [i for i in actual_indices if h_dim + l_dim + c_dim <= i < h_dim + l_dim + c_dim + g_dim]

    mapping = {idx: i for i, idx in enumerate(actual_indices)}
    h_imp = [importances[mapping[i]] for i in h_idx if i in mapping]
    l_imp = [importances[mapping[i]] for i in l_idx if i in mapping]
    c_imp = [importances[mapping[i]] for i in c_idx if i in mapping]
    g_imp = [importances[mapping[i]] for i in g_idx if i in mapping]

    groups = []
    vals = []
    if h_imp: groups.append("HOG"); vals.append(np.sum(h_imp))
    if l_imp: groups.append("LBP"); vals.append(np.sum(l_imp))
    if c_imp: groups.append("Renk Hist."); vals.append(np.sum(c_imp))
    if g_imp: groups.append("Gabor"); vals.append(np.sum(g_imp))

    if not vals:
        return

    vals = np.array(vals)
    vals = vals / np.sum(vals) * 100

    plt.figure(figsize=(7, 5))
    colors = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b"][:len(groups)]
    bars = plt.bar(groups, vals, color=colors, edgecolor="gray", width=0.5)
    plt.title(f"Öznitelik Grubu Önem Derecesi (RF) — {fs_name}")
    plt.ylabel("Toplam Katkı Oranı (%)")
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"%{yval:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / f"rf_importance_{_safe(fs_name)}.png", dpi=300)
    plt.close()


def plot_overall_comparison(df_results: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    df_melt = df_results.melt(
        id_vars=["Öznitelik Seti", "Sınıflandırıcı"],
        value_vars=["Val Doğruluğu", "Test Doğruluğu"],
        var_name="Veri Seti", value_name="Doğruluk"
    )
    df_melt["Doğruluk"] *= 100

    g = sns.catplot(
        data=df_melt, kind="bar",
        x="Öznitelik Seti", y="Doğruluk", hue="Sınıflandırıcı", col="Veri Seti",
        palette="muted", alpha=0.9, height=6, aspect=1.1
    )
    g.set_axis_labels("Öznitelik Seçim Yöntemi", "Doğruluk Oranı (%)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(70, 90)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.text(p.get_x() + p.get_width()/2., p.get_height() + 0.3,
                        f"{p.get_height():.1f}", ha="center", fontsize=9, fontweight='bold')
    
    plt.savefig(outdir / "results_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(all_f1: dict, outdir: Path):
    combos = [f"{k[0]}\n+{k[1]}" for k in all_f1.keys()]
    matrix = np.array(list(all_f1.values()))

    df = pd.DataFrame(matrix, index=combos, columns=SHORT_NAMES)

    plt.figure(figsize=(14, 10))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Tüm Kombinasyonlar İçin Sınıf Bazlı F1-Skor Dağılımı")
    plt.xlabel("Sınıf")
    plt.ylabel("Kombinasyon (Seçim Yöntemi + Model)")
    plt.tight_layout()
    plt.savefig(outdir / "per_class_f1_all.png", dpi=300)
    plt.close()


def plot_weak_class_comparison(all_f1: dict, outdir: Path):
    records = []
    for (fs, clf), f1s in all_f1.items():
        for cn, f1 in zip(CLASS_NAMES, f1s):
            if cn in WEAK_CLASSES:
                records.append({
                    "Kombinasyon": f"{fs} + {clf}",
                    "Sınıf": cn,
                    "F1-Skoru": f1
                })
    df = pd.DataFrame(records)
    if df.empty: return

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x="Kombinasyon", y="F1-Skoru", hue="Sınıf", palette="Set2")
    plt.xticks(rotation=45, ha="right")
    plt.title("Zorlu Sınıfların Kombinasyonlara Göre F1 Başarı Karşılaştırması")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(outdir / "weak_class_comparison.png", dpi=300)
    plt.close()


def plot_feature_set_heatmap(all_f1: dict, outdir: Path):
    fs_class = {}
    for (fs, clf), f1s in all_f1.items():
        if fs not in fs_class:
            fs_class[fs] = np.zeros(len(CLASS_NAMES))
        fs_class[fs] += np.array(f1s)

    for fs in fs_class:
        count = sum(1 for k in all_f1.keys() if k[0] == fs)
        if count > 0:
            fs_class[fs] /= count

    df = pd.DataFrame(fs_class, index=SHORT_NAMES).transpose()
    fs_order = ["Tüm Öznitelikler", "Pearson", "CFS", "LASSO", "PSO"]
    df = df.reindex(fs_order)

    plt.figure(figsize=(12, 5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="Purples", linewidths=0.7)
    plt.title("Öznitelik Seçim Yöntemlerinin Sınıf Bazlı Ortalama F1 Etkisi")
    plt.xlabel("Sınıf")
    plt.ylabel("Öznitelik Seti")
    plt.tight_layout()
    plt.savefig(outdir / "feature_set_class_heatmap.png", dpi=300)
    plt.close()


def plot_classifier_comparison_radar(all_f1: dict, outdir: Path):
    clf_f1 = {}
    for (fs, clf), f1s in all_f1.items():
        if clf not in clf_f1:
            clf_f1[clf] = np.zeros(len(CLASS_NAMES))
        clf_f1[clf] += np.array(f1s)

    for clf in clf_f1:
        count = sum(1 for k in all_f1.keys() if k[1] == clf)
        if count > 0:
            clf_f1[clf] /= count

    categories = SHORT_NAMES
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=11)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.1)

    colors = {'SVM': '#ff7f0e', 'RandomForest': '#2ca02c', 'LightGBM': '#1f77b4'}

    for clf, values in clf_f1.items():
        val_list = values.tolist()
        val_list += val_list[:1]
        ax.plot(angles, val_list, linewidth=2, linestyle='solid', label=clf, color=colors.get(clf))
        ax.fill(angles, val_list, color=colors.get(clf), alpha=0.1)

    plt.title("Sınıflandırıcıların Sınıf Bazlı Model Karşılaştırma Grafiği (Radar)", size=14, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(outdir / "radar_comparison.png", dpi=300)
    plt.close()


def plot_precision_recall_f1_heatmap(all_reports: dict, outdir: Path):
    best_combo = None
    best_acc = -1
    for k, v in all_reports.items():
        if v['test_acc'] > best_acc:
            best_acc = v['test_acc']
            best_combo = k

    if best_combo is None: return
    df_rep = all_reports[best_combo]['report']

    df_filtered = df_rep.loc[CLASS_NAMES, ['precision', 'recall', 'f1-score']]

    plt.figure(figsize=(8, 7))
    sns.heatmap(df_filtered, annot=True, fmt=".3f", cmap="YlOrBr", linewidths=0.5)
    plt.title(f"En İyi Model Metrik Kırılımı\n({best_combo[0]} + {best_combo[1]})")
    plt.xlabel("Metrik")
    plt.ylabel("Doku Sınıfı")
    plt.tight_layout()
    plt.savefig(outdir / "best_combo_metrics_heatmap.png", dpi=300)
    plt.close()


def plot_timing_analysis(df_timing: pd.DataFrame, outdir: Path):
    df_sorted = df_timing.sort_values(by="Süre (s)", ascending=True)

    plt.figure(figsize=(11, 6))
    bars = plt.barh(df_sorted["Kombinasyon"], df_sorted["Süre (s)"], color="#475569", edgecolor="none", height=0.6)
    plt.title("Kombinasyonların Hesaplama Süresi Analizi (Eğitim Süreleri)")
    plt.xlabel("Süre (Saniye)")
    plt.ylabel("Kombinasyon")
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(1.0, width*0.02), bar.get_y() + bar.get_height()/2, f"{width:.1f}s",
                 ha='left', va='center', fontsize=9, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    plt.savefig(outdir / "timing_chart.png", dpi=300)
    plt.close()


def _safe(s: str) -> str:
    return s.replace(" ", "_").replace(".", "").replace("ü", "u").replace("Ö", "O").replace("ö", "o")


def main():
    args = parse_args()

    features_dir = Path(args.features_dir)
    outdir       = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cm_dir = outdir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    model_dir = outdir.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Boyut bilgileri (RF grubu için)
    hog_dim, lbp_dim, rgb_dim, gabor_dim = 144, 256, 96, 64

    feature_sets = ["Tüm Öznitelikler", "Pearson", "CFS", "LASSO", "PSO"]
    classifiers  = ALL_CLASSIFIERS if args.clf == 'all' else [args.clf]

    results = {}
    detailed_results = []
    timing_records = []
    all_f1 = {}
    all_reports = {}

    print("PathMNIST Sınıflandırma Boru Hattı Başlatılıyor...")
    print(f"Öznitelik dizini: {features_dir}")
    print(f"Çıktı dizini:    {outdir}\n")

    for fs_name in feature_sets:
        print(f"■ {fs_name} yükleniyor...")
        try:
            X_data, y_data, feat_idx = load_dataset(features_dir, _safe(fs_name))
        except Exception as e:
            print(f"  Hata (Veri seti yüklenemedi): {e}. Atlanıyor.")
            continue

        # Veriyi bölme (PathMNIST standartlarına göre)
        # Orijinal oranlar: 89996 / 10015 / 7180
        # Kodun dinamik kalması için yüklü verinin şekline göre dilimliyoruz
        n_total = len(X_data)
        if n_total == 115965 + 10015 + 7180: # Augmentation dahil durum
            n_train = 115965
            n_val   = 10015
        else:
            # Standart oranlara göre dinamik paylaştırma
            n_train = int(n_total * (89996 / 107191))
            n_val   = int(n_total * (10015 / 107191))

        X_train, y_train = X_data[:n_train], y_data[:n_train]
        X_val,   y_val   = X_data[n_train:n_train+n_val], y_data[n_train:n_train+n_val]
        X_test,  y_test  = X_data[n_train+n_val:], y_data[n_train+n_val:]

        for clf_name in classifiers:
            clf = get_classifier(clf_name, args.seed)

            # Sadece SVM'in çok uzun sürmemesi için örnek sınırlandırma koruması (isteğe bağlı)
            Xtr, ytr = X_train, y_train
            Xvl, yvl = X_val, y_val
            Xte, yte = X_test, y_test

            test_acc, val_acc, y_pred, elapsed = train_evaluate(
                clf, Xtr, ytr, Xvl, yvl, Xte, yte
            )
            results[(fs_name, clf_name)] = test_acc

            detailed_results.append({
                "Öznitelik Seti":  fs_name,
                "Sınıflandırıcı": clf_name,
                "Val Doğruluğu":  val_acc,
                "Test Doğruluğu": test_acc,
                "Süre (s)":       elapsed,
                "Feature_Count":  Xtr.shape[1],
            })
            timing_records.append({
                "Kombinasyon":   f"{fs_name} + {clf_name}",
                "Süre (s)":      elapsed,
                "Feature_Count": Xtr.shape[1],
            })

            print(f"  {clf_name:20s} -> Val: {val_acc:.4f}  Test: {test_acc:.4f}  ({elapsed:.1f}s)")

            if y_pred is not None:
                if GRAFIKLERI_CIZ:
                    save_confusion_matrix(y_test, y_pred, fs_name, clf_name, cm_dir)
                
                report_df = save_per_class_report(y_test, y_pred, fs_name, clf_name, outdir)

                f1_scores = [
                    report_df.loc[cn, 'f1-score'] if cn in report_df.index else 0.0
                    for cn in CLASS_NAMES
                ]
                all_f1[(fs_name, clf_name)]     = f1_scores
                all_reports[(fs_name, clf_name)] = {
                    'report':   report_df,
                    'test_acc': test_acc,
                    'val_acc':  val_acc,
                }

                if GRAFIKLERI_CIZ and clf_name == 'RandomForest':
                    plot_rf_feature_importance(
                        clf, feat_idx,
                        hog_dim, lbp_dim, rgb_dim, gabor_dim,
                        outdir, fs_name
                    )

            with open(model_dir / f"model_{_safe(fs_name)}_{_safe(clf_name)}.pkl", "wb") as f:
                pickle.dump(clf, f)

    print("\nTüm kombinasyonlar tamamlandı.")

    # --- Sonuçları kaydet ---
    df_results = pd.DataFrame(detailed_results)
    df_timing  = pd.DataFrame(timing_records)

    results_path = outdir / "results.csv"
    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        df_results  = pd.concat([df_existing, df_results], ignore_index=True).drop_duplicates(
            subset=["Öznitelik Seti", "Sınıflandırıcı"], keep='last'
        )
    df_results.to_csv(results_path, index=False)

    timing_path = outdir / "timing.csv"
    if timing_path.exists():
        df_existing_t = pd.read_csv(timing_path)
        df_timing     = pd.concat([df_existing_t, df_timing], ignore_index=True).drop_duplicates(
            subset=["Kombinasyon"], keep='last'
        )
    df_timing.to_csv(timing_path, index=False)

    # --- Tablolar ---
    fs_order  = ["Tüm Öznitelikler", "Pearson", "CFS", "LASSO", "PSO"]
    clf_order = ["SVM", "RandomForest", "LightGBM"]

    if len(df_results) > 0:
        try:
            pivot_test = df_results.pivot(
                index="Öznitelik Seti", columns="Sınıflandırıcı", values="Test Doğruluğu"
            ).reindex(fs_order).reindex(
                columns=[c for c in clf_order if c in df_results["Sınıflandırıcı"].unique()]
            )
            pivot_val = df_results.pivot(
                index="Öznitelik Seti", columns="Sınıflandırıcı", values="Val Doğruluğu"
            ).reindex(fs_order).reindex(
                columns=[c for c in clf_order if c in df_results["Sınıflandırıcı"].unique()]
            )

            print("\nTest Doğruluğu (%):\n")
            print((pivot_test * 100).round(2).to_string())
            print("\nVal Doğruluğu (%):\n")
            print((pivot_val * 100).round(2).to_string())

            if GRAFIKLERI_CIZ:
                plt.figure(figsize=(8, 6))
                sns.heatmap(pivot_test * 100, annot=True, fmt=".2f",
                            cmap="YlGnBu", linewidths=0.5, linecolor="gray")
                plt.title("PathMNIST – Test Doğruluğu (%)")
                plt.xlabel("Sınıflandırıcı"); plt.ylabel("Öznitelik Seçim Yöntemi")
                plt.tight_layout()
                plt.savefig(outdir / "heatmap_test.png", dpi=300, bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(8, 6))
                sns.heatmap(pivot_val * 100, annot=True, fmt=".2f",
                            cmap="YlOrRd", linewidths=0.5, linecolor="gray")
                plt.title("PathMNIST – Val Doğruluğu (%)")
                plt.xlabel("Sınıflandırıcı"); plt.ylabel("Öznitelik Seçim Yöntemi")
                plt.tight_layout()
                plt.savefig(outdir / "heatmap_val.png", dpi=300, bbox_inches="tight")
                plt.close()

                plot_overall_comparison(df_results, outdir)

        except Exception as e:
            print(f"Tablo/grafik hatası: {e}")

    # --- Gelişmiş grafikler ---
    if GRAFIKLERI_CIZ and all_f1:
        print("\nGelişmiş grafikler oluşturuluyor...")
        plot_per_class_f1(all_f1, outdir)
        plot_weak_class_comparison(all_f1, outdir)
        plot_feature_set_heatmap(all_f1, outdir)
        plot_classifier_comparison_radar(all_f1, outdir)

    if GRAFIKLERI_CIZ and all_reports:
        plot_precision_recall_f1_heatmap(all_reports, outdir)

    if GRAFIKLERI_CIZ and len(df_timing) > 0:
        plot_timing_analysis(df_timing, outdir)

    if results:
        best_key = max(results, key=results.get)
        print(f"\n🏆 En iyi kombinasyon: {best_key[0]} + {best_key[1]} -> {results[best_key]:.4f}")

    print(f"\nTüm çıktılar: {outdir}")
    print(f"Modeller: {model_dir}")
    print(f"Confusion matrix'ler: {cm_dir}")


if __name__ == "__main__":
    main()
