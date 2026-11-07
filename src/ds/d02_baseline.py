"""DS d02: naive k-means baseline, the benchmark the curated candidates (d03) must beat.

"Naive" = the d01 model columns as stored: median-imputed, standard-scaled, categoricals one-hot encoded.
No log transforms, winsorising, price levels or correlation pruning.

Sweep k = K_RANGE. For each k (nested MLflow run): native metrics, Gower silhouette in the curated fv1
space (common yardstick), segment sizes and bootstrap stability. The reference baseline k is the admissible
k (no segment under 2%, stability >= 0.80) with the best Gower silhouette; its model is logged and its labels
saved as candidate `baseline-kmeans-k<k>`. The answer key is not used.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from config import DS_REPORTS_DIR, FEATURE_VERSION, RANDOM_SEED  # noqa: E402
from src.ds import tracking  # noqa: E402
from src.ds.candidates import load_features, save_candidate, segment_outcomes  # noqa: E402
from src.ds.evaluate import (MIN_SEGMENT_SHARE, STABILITY_TARGET, common_metrics, native_metrics,  # noqa: E402
                             size_stats, stability)
from src.ds.feature_spec import MODEL_FEATURES  # noqa: E402
from src.ds.preprocess import SegmentationPreprocessor  # noqa: E402

K_RANGE = range(5, 13)
N_INIT = 10
N_INIT_BOOT = 3
FEATURE_SPACE = "raw-onehot"

NUM_COLS = [f.name for f in MODEL_FEATURES if f.kind != "cat"]
CAT_COLS = [f.name for f in MODEL_FEATURES if f.kind == "cat"]


def naive_prep() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_COLS),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Unknown"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CAT_COLS),
    ])


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUM_COLS + CAT_COLS].copy()
    X[NUM_COLS] = X[NUM_COLS].astype(float)
    X[CAT_COLS] = X[CAT_COLS].astype(object).where(X[CAT_COLS].notna(), np.nan)
    return X


def plot_sweep(sweep: pd.DataFrame, chosen_k: int, path) -> None:
    panels = [("inertia", "Inertia (elbow)"), ("silhouette_gower", "Gower silhouette (common yardstick)"),
              ("calinski_harabasz", "Calinski-Harabasz"), ("stability_ari_mean", "Bootstrap stability (ARI)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(sweep.k, sweep[col], marker="o", color="#4C6EF5")
        ax.axvline(chosen_k, color="#E8590C", ls="--", lw=1)
        if col == "stability_ari_mean":
            ax.axhline(STABILITY_TARGET, color="grey", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
    fig.suptitle(f"Naive k-means baseline sweep (reference k = {chosen_k})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    df, train = load_features()
    X_all = model_input(df)
    X_train = X_all[train].reset_index(drop=True)

    prep = naive_prep().fit(X_train)
    Z_train = prep.transform(X_train)
    curated = SegmentationPreprocessor().fit(df[train])
    C_train = curated.transform(df[train]).reset_index(drop=True)
    print(f"Baseline space: {Z_train.shape[1]} columns ({len(NUM_COLS)} numeric + one-hot); "
          f"{len(X_train):,} training customers")

    tracking.setup()
    rows, models = [], {}
    with mlflow.start_run(run_name="d02-baseline") as parent:
        mlflow.set_tags({"step": "d02", "feature_version": FEATURE_VERSION, "feature_space": FEATURE_SPACE})
        mlflow.log_params({"model": "kmeans", "k_range": f"{K_RANGE.start}-{K_RANGE.stop - 1}", "n_init": N_INIT,
                           "n_train": len(X_train), "n_columns": Z_train.shape[1], "seed": RANDOM_SEED})
        for k in K_RANGE:
            with mlflow.start_run(run_name=f"baseline-kmeans-k{k}", nested=True):
                km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_SEED).fit(Z_train)
                labels = km.labels_

                def refit(idx, seed, k=k):
                    return KMeans(n_clusters=k, n_init=N_INIT_BOOT, random_state=seed).fit(Z_train[idx]).predict(Z_train)

                m = {"k": k, "inertia": float(km.inertia_), **size_stats(labels), **native_metrics(Z_train, labels),
                     **common_metrics(C_train, labels, curated.num_cols_, curated.cat_cols_),
                     **stability(refit, len(Z_train), labels)}
                mlflow.log_params({"k": k, "model": "kmeans", "feature_space": FEATURE_SPACE})
                mlflow.log_metrics({key: v for key, v in m.items() if key != "k"})
                models[k] = km
                rows.append(m)
                print(f"  k={k:>2}  gower_sil={m['silhouette_gower']:.3f}  CH={m['calinski_harabasz']:,.0f}  "
                      f"DB={m['davies_bouldin']:.2f}  stab={m['stability_ari_mean']:.2f}  min_share={m['min_share']:.1%}")

        sweep = pd.DataFrame(rows)
        sweep["admissible"] = (sweep.n_small_segments == 0) & (sweep.stability_ari_mean >= STABILITY_TARGET)
        pool = sweep[sweep.admissible] if sweep.admissible.any() else sweep
        best_k = int(pool.loc[pool.silhouette_gower.idxmax(), "k"])

        # Reference baseline: full pipeline, labels for every customer (probable duplicates via predict)
        pipe = Pipeline([("prep", prep), ("kmeans", models[best_k])])
        labels_all = pipe.predict(X_all)
        prof = segment_outcomes(df.customer_id, labels_all)

        DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(DS_REPORTS_DIR / "d02_baseline_sweep.csv", index=False)
        prof.to_csv(DS_REPORTS_DIR / "d02_baseline_segments.csv")
        plot_sweep(sweep, best_k, DS_REPORTS_DIR / "d02_baseline_sweep.png")
        for name in ("d02_baseline_sweep.csv", "d02_baseline_segments.csv", "d02_baseline_sweep.png"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / name))
        best = sweep.set_index("k").loc[best_k]
        mlflow.log_params({"reference_k": best_k, "reference_admissible": bool(best.admissible)})
        mlflow.log_metrics({f"reference_{c}": float(best[c]) for c in
                            ("silhouette_gower", "stability_ari_mean", "min_share", "calinski_harabasz")})
        # skops is MLflow's safe serialiser; numpy.dtype comes from the one-hot encoder we built here
        mlflow.sklearn.log_model(pipe, name="baseline_kmeans", skops_trusted_types=["numpy.dtype"],
                                 signature=infer_signature(X_all.head(200), labels_all[:200]))

        candidate_id = f"baseline-kmeans-k{best_k}"
        metrics = {c: float(best[c]) for c in ("silhouette_gower", "silhouette_native", "calinski_harabasz",
                                               "davies_bouldin", "stability_ari_mean", "min_share", "max_share")}
        save_candidate(candidate_id, "kmeans", best_k, FEATURE_SPACE, parent.info.run_id, metrics,
                       df.customer_id, labels_all)

    print(f"\nReference baseline: k={best_k} (admissible: {bool(best.admissible)}; "
          f"rule = no segment < {MIN_SEGMENT_SHARE:.0%}, stability >= {STABILITY_TARGET}, best Gower silhouette)")
    print(prof.to_string())
    print(f"Saved candidate {candidate_id}; MLflow run {parent.info.run_id}")


if __name__ == "__main__":
    main()
