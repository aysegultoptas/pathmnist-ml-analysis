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

import medmnist
from medmnist import PathMNIST

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.filters import gabor
from skimage.transform import rotate
from skimage import exposure

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import pearsonr
from pyswarms.discrete import BinaryPSO

warnings.filterwarnings("ignore")

# ==============================================================================
# GRAFİKLERİ ÇİZDİRMEK İSTİYOR MUSUNUZ? 
# GitHub için varsayılan olarak False (Kapalı). İsteyen True (Açık) yapabilir.
GRAFIKLERI_CIZ = False  
# ==============================================================================

TEST_MODE    = False
RANDOM_STATE = 42

CLASS_NAMES = [
    'Adipose', 'Background', 'Debris', 'Lymphocytes',
    'Mucus', 'Smooth Muscle', 'Normal Colon Mucosa',
    'Cancer-Assoc. Stroma', 'Colorectal Adenocarcinoma'
]

CFS_MIN_FEATURES = 50
TARGET_PER_CLASS = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PathMNIST – Öznitelik Çıkarımı ve Seçimi")
    p.add_argument("--test-mode", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs/features", help="Kaydedilecek klasör")
    p.add_argument("--download", action="store_true")
    p.add_argument("--data-root", type=str, default=os.environ.get("MEDMNIST_ROOT", ""))
    p.add_argument("--no-augment", action="store_true", help="Augmentation'ı devre dışı bırak")
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_set_medmnist_root(data_root: str) -> None:
    if data_root:
        os.environ["MEDMNIST_ROOT"] = str(Path(data_root).expanduser().resolve())


def dataset_to_numpy(dataset):
    imgs   = dataset.imgs
    labels = dataset.labels.ravel()
    return imgs, labels


def plot_class_samples(X_train_raw, y_train, outdir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for cls_idx in range(9):
        mask   = y_train == cls_idx
        sample = X_train_raw[mask][0]
        ax     = axes[cls_idx // 3, cls_idx % 3]
        ax.imshow(sample)
        ax.set_title(f"{cls_idx}: {CLASS_NAMES[cls_idx]}", fontsize=7)
        ax.axis("off")
    plt.suptitle("PathMNIST – Her Sınıftan Örnek", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "class_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(y_before, y_after, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts_before = [np.sum(y_before == i) for i in range(9)]
    counts_after  = [np.sum(y_after  == i) for i in range(9)]

    axes[0].bar(range(9), counts_before, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(9))
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    axes[0].set_title("Augmentation Öncesi Sınıf Dağılımı")
    axes[0].set_ylabel("Örnek Sayısı")

    axes[1].bar(range(9), counts_after, color='forestgreen', alpha=0.8)
    axes[1].set_xticks(range(9))
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    axes[1].set_title("Augmentation Sonrası Sınıf Dağılımı")
    axes[1].set_ylabel("Örnek Sayısı")

    plt.suptitle("Sınıf Dağılımı Karşılaştırması", fontsize=12)
    plt.tight_layout()
    fig.savefig(outdir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    angle = rng.choice([0, 90, 180, 270])
    if angle != 0:
        img = rotate(img, angle, preserve_range=True)
    if rng.random() > 0.5:
        img = np.fliplr(img)
    if rng.random() > 0.5:
        img = np.flipud(img)
    factor = 1.0 + rng.uniform(-0.2, 0.2)
    img = np.clip(img * factor, 0, 1)
    if rng.random() > 0.5:
        img = exposure.adjust_gamma(img, gamma=rng.uniform(0.8, 1.2))
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * (1.0 + rng.uniform(-0.1, 0.1)), 0, 1)
    return (img * 255).astype(np.uint8)


def balance_with_augmentation(X: np.ndarray, y: np.ndarray, target_per_class: int, rng: np.random.Generator, test_mode: bool = False) -> tuple[np.ndarray, np.ndarray]:
    X_aug_list = [X]
    y_aug_list = [y]
    classes      = np.unique(y)
    class_counts = {c: np.sum(y == c) for c in classes}

    print(f"\nAugmentation hedefi: {target_per_class} örnek/sınıf")
    for cls in classes:
        current = class_counts[cls]
        needed  = target_per_class - current
        if needed <= 0:
            continue
        cls_indices = np.where(y == cls)[0]
        X_cls       = X[cls_indices]
        augmented = []
        while len(augmented) < needed:
            src_idx = rng.integers(0, len(X_cls))
            aug_img = augment_image(X_cls[src_idx], rng)
            augmented.append(aug_img)
        X_aug_list.append(np.array(augmented))
        y_aug_list.append(np.full(len(augmented), cls, dtype=y.dtype))

    X_balanced = np.concatenate(X_aug_list, axis=0)
    y_balanced = np.concatenate(y_aug_list, axis=0)
    shuffle_idx = rng.permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


def extract_hog_features(images):
    features = []
    for img in images:
        gray = rgb2gray(img)
        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        features.append(feat)
    return np.array(features)


def extract_lbp_features(images, P=8, R=1, n_bins=256):
    features = []
    for img in images:
        gray = rgb2gray(img)
        lbp  = local_binary_pattern(gray, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.append(hist)
    return np.array(features)


def extract_color_histogram_features(images, n_bins=32):
    features = []
    for img in images:
        img_float = img.astype(np.float32) / 255.0
        hist_r, _ = np.histogram(img_float[:, :, 0], bins=n_bins, range=(0, 1), density=True)
        hist_g, _ = np.histogram(img_float[:, :, 1], bins=n_bins, range=(0, 1), density=True)
        hist_b, _ = np.histogram(img_float[:, :, 2], bins=n_bins, range=(0, 1), density=True)
        features.append(np.concatenate([hist_r, hist_g, hist_b]))
    return np.array(features)


def extract_gabor_features(images):
    frequencies = [0.1, 0.2, 0.3, 0.4]
    thetas = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]
    features = []
    for img in images:
        gray     = rgb2gray(img)
        feat_vec = []
        for freq in frequencies:
            for theta in thetas:
                filt_real, filt_imag = gabor(gray, frequency=freq, theta=theta)
                magnitude = np.sqrt(filt_real**2 + filt_imag**2)
                feat_vec.extend([magnitude.mean(), magnitude.std()])
        features.append(feat_vec)
    return np.array(features)


def pearson_feature_selection(X, y):
    correlations = []
    for i in range(X.shape[1]):
        try:
            corr, _ = pearsonr(X[:, i], y)
            if np.isnan(corr): corr = 0
        except:
            corr = 0
        correlations.append(abs(corr))
    correlations  = np.array(correlations)
    threshold     = np.percentile(correlations, 75)
    selected_idx  = np.where(correlations > threshold)[0]
    return selected_idx, correlations


def cfs_feature_selection(X, y, pearson_scores=None):
    n_features = X.shape[1]
    r_cf = np.array([abs(pearsonr(X[:, i], y)[0]) if len(np.unique(X[:, i])) > 1 else 0 for i in range(n_features)])
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix)
    np.fill_diagonal(corr_matrix, 0)
    abs_corr = np.abs(corr_matrix)

    selected           = []
    remaining          = list(range(n_features))
    best_overall_merit = -np.inf
    MERIT_TOLERANCE    = 0.005

    while len(remaining) > 0:
        best_feat  = None
        best_merit = -np.inf
        for feat in remaining:
            candidate  = selected + [feat]
            k          = len(candidate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_r_cf  = np.mean(r_cf[candidate])
            if k == 1:
                mean_r_ff = 0
            else:
                sub_matrix = abs_corr[np.ix_(candidate, candidate)]
                mean_r_ff  = sub_matrix.sum() / (k * (k - 1))
            denom = np.sqrt(k + k * (k - 1) * mean_r_ff + 1e-8)
            merit = (k * mean_r_cf) / denom
            if merit > best_merit:
                best_merit = merit
                best_feat  = feat

        if best_merit <= best_overall_merit - MERIT_TOLERANCE:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_overall_merit = max(best_overall_merit, best_merit)

    selected = np.array(selected)
    if len(selected) < CFS_MIN_FEATURES:
        scores   = pearson_scores if pearson_scores is not None else r_cf
        selected = np.argsort(scores)[::-1][:CFS_MIN_FEATURES]
    return selected


def lasso_feature_selection(X_train, y_train, X_val, X_test, C=0.01):
    lasso_clf = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=500, random_state=42)
    lasso_clf.fit(X_train, y_train)
    coef         = lasso_clf.coef_
    coef_sum     = np.sum(np.abs(coef), axis=0)
    selected_idx = np.where(coef_sum > 0)[0]
    return selected_idx, lasso_clf


def pso_fitness(position, X_train, y_train):
    n_particles = position.shape[0]
    cost        = np.zeros(n_particles)
    for i in range(n_particles):
        mask = position[i].astype(bool)
        if np.sum(mask) == 0:
            cost[i] = 1.0
            continue
        X_sub = X_train[:, mask]
        clf = RandomForestClassifier(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=42, class_weight='balanced')
        try:
            scores  = cross_val_score(clf, X_sub, y_train, cv=3, scoring='f1_macro')
            cost[i] = 1.0 - scores.mean()
        except:
            cost[i] = 1.0
    return cost


def main() -> None:
    args = parse_args()
    global TEST_MODE, RANDOM_STATE
    TEST_MODE    = bool(args.test_mode)
    RANDOM_STATE = int(args.seed)
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    outdir = ensure_dir(args.outdir)
    maybe_set_medmnist_root(args.data_root)

    train_dataset = PathMNIST(split="train", download=bool(args.download))
    val_dataset   = PathMNIST(split="val",   download=bool(args.download))
    test_dataset  = PathMNIST(split="test",  download=bool(args.download))

    X_train_raw, y_train = dataset_to_numpy(train_dataset)
    X_val_raw,   y_val   = dataset_to_numpy(val_dataset)
    X_test_raw,  y_test  = dataset_to_numpy(test_dataset)

    if TEST_MODE:
        X_train_raw, y_train = X_train_raw[:4000], y_train[:4000]
        X_val_raw,   y_val   = X_val_raw[:500],    y_val[:500]
        X_test_raw,  y_test  = X_test_raw[:500],   y_test[:500]

    if GRAFIKLERI_CIZ:
        plot_class_samples(X_train_raw, y_train, outdir)

    if not args.no_augment:
        class_counts     = {c: int(np.sum(y_train == c)) for c in np.unique(y_train)}
        target_per_class = max(class_counts.values())
        X_train_aug, y_train_aug = balance_with_augmentation(X_train_raw, y_train, target_per_class=target_per_class, rng=rng, test_mode=TEST_MODE)
        if GRAFIKLERI_CIZ:
            plot_class_distribution(y_train, y_train_aug, outdir)
        X_train_raw, y_train = X_train_aug, y_train_aug

    print("\nÖznitelikler çıkarılıyor...")
    hog_train = extract_hog_features(X_train_raw);     hog_val = extract_hog_features(X_val_raw);     hog_test = extract_hog_features(X_test_raw)
    lbp_train = extract_lbp_features(X_train_raw);     lbp_val = extract_lbp_features(X_val_raw);     lbp_test = extract_lbp_features(X_test_raw)
    color_train = extract_color_histogram_features(X_train_raw); color_val = extract_color_histogram_features(X_val_raw); color_test = extract_color_histogram_features(X_test_raw)
    gabor_train = extract_gabor_features(X_train_raw); gabor_val = extract_gabor_features(X_val_raw); gabor_test = extract_gabor_features(X_test_raw)

    X_train_full = np.hstack([hog_train, lbp_train, color_train, gabor_train])
    X_val_full   = np.hstack([hog_val,   lbp_val,   color_val,   gabor_val])
    X_test_full  = np.hstack([hog_test,  lbp_test,  color_test,  gabor_test])

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_val_scaled   = scaler.transform(X_val_full)
    X_test_scaled  = scaler.transform(X_test_full)

    with open(outdir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.savez_compressed(outdir / "X_Tum_Oznitelikler_scaled.npz", X=X_train_scaled, y=y_train)
    # classify.py için standart adlandırma uyumluluğu amaçlı train/val/test ayrı ayrı kaydedilir
    np.savez_compressed(outdir / "X_train_scaled.npz", X=X_train_scaled, y=y_train)
    np.savez_compressed(outdir / "X_val_scaled.npz", X=X_val_scaled, y=y_val)
    np.savez_compressed(outdir / "X_test_scaled.npz", X=X_test_scaled, y=y_test)

    print("\nÖznitelik Seçimleri Hesaplanıyor...")
    pearson_idx, pearson_scores = pearson_feature_selection(X_train_scaled, y_train)
    cfs_idx = cfs_feature_selection(X_train_scaled, y_train, pearson_scores=pearson_scores)
    lasso_idx, _ = lasso_feature_selection(X_train_scaled, y_train, X_val_scaled, X_test_scaled)

    PSO_N_FEATURES  = 100 if not TEST_MODE else 50
    PSO_ITERS       = 40  if not TEST_MODE else 1
    PSO_N_PARTICLES = 20  if not TEST_MODE else 5
    pre_pso_idx = np.argsort(pearson_scores)[::-1][:PSO_N_FEATURES]
    
    optimizer = BinaryPSO(n_particles=PSO_N_PARTICLES, dimensions=PSO_N_FEATURES, options={"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 3, "p": 2})
    best_cost, best_pos = optimizer.optimize(lambda pos: pso_fitness(pos, X_train_scaled[:, pre_pso_idx], y_train), iters=PSO_ITERS, verbose=False)
    pso_idx = pre_pso_idx[np.where(best_pos.astype(bool))[0]]

    np.savez_compressed(
        outdir / "feature_indices.npz",
        all_features   = np.arange(X_train_scaled.shape[1]),
        pearson        = pearson_idx,
        pearson_scores = pearson_scores,
        cfs            = cfs_idx,
        lasso          = lasso_idx,
        pso            = pso_idx,
        hog_dim        = np.array([hog_train.shape[1]]),
        lbp_dim        = np.array([lbp_train.shape[1]]),
        color_dim      = np.array([color_train.shape[1]]),
        gabor_dim      = np.array([gabor_train.shape[1]]),
    )
    print("\n[BAŞARILI] Tüm indeksler ve veriler 'outputs/features' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
