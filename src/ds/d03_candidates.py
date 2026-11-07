"""DS d03: curated candidate segmentations on the fv1 feature space.

Families (each its own MLflow parent run with a nested run per k):
    kmeans  k-means on scaled numeric + one-hot categoricals
    kproto  k-prototypes on scaled numeric + native categoricals (fit on a stratified sample, predict the rest)
    gmm     Gaussian mixture (diagonal covariance) on the k-means encoding; also logs BIC

Every family uses categorical emphasis e = 1 (each variable has equal expected weight; see segmenter.py), the
same k sweep and the same scorecard as the d02 baseline. The reference k per family follows the d02 rule:
no segment < 2%, stability >= 0.80, then best Gower silhouette. At the reference k, e = 0.5 and e = 2 are refitted
as a sensitivity check (agreement ARI with e = 1); they are reported, not used to choose.
The answer key is not used.

Run all families:   python run_pipeline.py candidates
Run some:           python -m src.ds.d03_candidates kproto gmm
"""
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kmodes.kprototypes import KPrototypes  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from config import DS_REPORTS_DIR, FEATURE_VERSION, RANDOM_SEED  # noqa: E402
from src.ds import tracking  # noqa: E402
from src.ds.candidates import load_features, save_candidate, segment_outcomes  # noqa: E402
from src.ds.evaluate import STABILITY_TARGET, common_metrics, native_metrics, size_stats, stability  # noqa: E402
from src.ds.preprocess import SegmentationPreprocessor  # noqa: E402
from src.ds.segmenter import Segmenter, SegmenterPyfunc, encode, mixed  # noqa: E402

K_RANGE = range(5, 13)
EMPHASIS = 1.0
SENSITIVITY_EMPHASIS = (0.5, 2.0)
KPROTO_SAMPLE = 30_000
KPROTO_BOOT_SAMPLE = 20_000
FEATURE_SPACE = "curated-fv1"


@dataclass
class Family:
    name: str
    description: str
    fit: Callable          # fit(data, k, seed, emphasis, n_init) -> fitted model
    n_init: int
    n_init_boot: int
    n_boot: int


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Positional indices of a sample stratified by intent x customer type (keeps small groups represented)."""
    strata = df.primary_intent.astype(str) + "|" + df.customer_type.astype(str)
    frac = n / len(df)
    idx = (pd.Series(np.arange(len(df)), index=df.index).groupby(strata.to_numpy())
           .apply(lambda s: s.sample(frac=frac, random_state=seed)))
    return np.sort(idx.to_numpy())


class Data:
    """Curated training data in both encodings, built once."""

    def __init__(self):
        self.df, train = load_features()
        self.train_df = self.df[train].reset_index(drop=True)
        self.pre = SegmentationPreprocessor().fit(self.train_df)
        self.C = self.pre.transform(self.train_df)
        self.encoded = {e: encode(self.C, self.pre, e) for e in (EMPHASIS, *SENSITIVITY_EMPHASIS)}
        self.mixed, self.cat_idx = mixed(self.C, self.pre)
        self.fit_sample = stratified_sample(self.train_df, KPROTO_SAMPLE, RANDOM_SEED)


def fit_kmeans(d: Data, k, seed, e, n_init, rows=None):
    Z = d.encoded[e] if rows is None else d.encoded[e][rows]
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)


def fit_gmm(d: Data, k, seed, e, n_init, rows=None):
    Z = d.encoded[e] if rows is None else d.encoded[e][rows]
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=n_init, reg_covar=1e-4,
                           random_state=seed).fit(Z)


def fit_kproto(d: Data, k, seed, e, n_init, rows=None):
    rows = d.fit_sample if rows is None else rows
    return KPrototypes(n_clusters=k, init="Cao", gamma=2.0 * e, n_init=n_init, max_iter=100,
                       random_state=seed, n_jobs=min(n_init, 4)).fit(d.mixed[rows], categorical=d.cat_idx)


FAMILIES = {
    "kmeans": Family("kmeans", "k-means, scaled numeric + one-hot", fit_kmeans, 10, 3, 10),
    "kproto": Family("kproto", "k-prototypes, scaled numeric + categorical", fit_kproto, 4, 2, 5),
    "gmm": Family("gmm", "Gaussian mixture (diag), scaled numeric + one-hot", fit_gmm, 3, 1, 10),
}


def predict(d: Data, fam: Family, model, e=EMPHASIS) -> np.ndarray:
    if fam.name == "kproto":
        return np.asarray(model.predict(d.mixed, categorical=d.cat_idx))
    return np.asarray(model.predict(d.encoded[e]))


def bootstrap_rows(d: Data, fam: Family) -> tuple[int, float]:
    n = len(d.train_df)
    return n, (KPROTO_BOOT_SAMPLE / n if fam.name == "kproto" else 0.8)


def plot_sweep(sweep: pd.DataFrame, fam: Family, chosen_k: int, path) -> None:
    panels = [("silhouette_gower", "Gower silhouette (common yardstick)"), ("silhouette_native", "Silhouette (encoded space)"),
              ("stability_ari_mean", "Bootstrap stability (ARI)"), ("min_share", "Smallest segment share")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(sweep.k, sweep[col], marker="o", color="#4C6EF5")
        ax.axvline(chosen_k, color="#E8590C", ls="--", lw=1)
        if col == "stability_ari_mean":
            ax.axhline(STABILITY_TARGET, color="grey", ls=":", lw=1)
        if col == "min_share":
            ax.axhline(0.02, color="grey", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
    fig.suptitle(f"{fam.description}: sweep (reference k = {chosen_k})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_family(d: Data, fam: Family) -> dict:
    print(f"\n--- {fam.name}: {fam.description} ---")
    Z_native = d.encoded[EMPHASIS]
    rows, models = [], {}
    with mlflow.start_run(run_name=f"d03-{fam.name}") as parent:
        mlflow.set_tags({"step": "d03", "family": fam.name, "feature_version": FEATURE_VERSION,
                         "feature_space": FEATURE_SPACE})
        mlflow.log_params({"model": fam.name, "k_range": f"{K_RANGE.start}-{K_RANGE.stop - 1}", "emphasis": EMPHASIS,
                           "n_init": fam.n_init, "n_boot": fam.n_boot, "n_train": len(d.train_df),
                           "fit_rows": KPROTO_SAMPLE if fam.name == "kproto" else len(d.train_df), "seed": RANDOM_SEED})
        for k in K_RANGE:
            with mlflow.start_run(run_name=f"{fam.name}-k{k}", nested=True):
                model = fam.fit(d, k, RANDOM_SEED, EMPHASIS, fam.n_init)
                labels = predict(d, fam, model)

                def refit(idx, seed, k=k):
                    return predict(d, fam, fam.fit(d, k, seed, EMPHASIS, fam.n_init_boot, rows=idx))

                n_rows, frac = bootstrap_rows(d, fam)
                m = {"k": k, **size_stats(labels), **native_metrics(Z_native, labels),
                     **common_metrics(d.C, labels, d.pre.num_cols_, d.pre.cat_cols_),
                     **stability(refit, n_rows, labels, n_boot=fam.n_boot, frac=frac)}
                if fam.name == "gmm":
                    m["bic"] = float(model.bic(Z_native))
                mlflow.log_params({"k": k, "model": fam.name, "emphasis": EMPHASIS})
                mlflow.log_metrics({key: v for key, v in m.items() if key != "k"})
                models[k] = model
                rows.append(m)
                print(f"  k={k:>2}  gower_sil={m['silhouette_gower']:.3f}  native_sil={m['silhouette_native']:.3f}  "
                      f"stab={m['stability_ari_mean']:.2f}  min_share={m['min_share']:.1%}", flush=True)

        sweep = pd.DataFrame(rows)
        sweep["admissible"] = (sweep.n_small_segments == 0) & (sweep.stability_ari_mean >= STABILITY_TARGET)
        pool = sweep[sweep.admissible] if sweep.admissible.any() else sweep
        best_k = int(pool.loc[pool.silhouette_gower.idxmax(), "k"])
        best = sweep.set_index("k").loc[best_k]
        labels_best = predict(d, fam, models[best_k])

        # Sensitivity to categorical emphasis at the reference k (reported, not used to choose)
        sens = {}
        for e in SENSITIVITY_EMPHASIS:
            alt = predict(d, fam, fam.fit(d, best_k, RANDOM_SEED, e, fam.n_init), e=e)
            sens[f"sensitivity_ari_emphasis_{e}"] = float(adjusted_rand_score(labels_best, alt))
        print(f"  reference k={best_k} (admissible {bool(best.admissible)}); emphasis sensitivity ARI: "
              + ", ".join(f"{k.split('_')[-1]}: {v:.2f}" for k, v in sens.items()))

        segmenter = Segmenter(d.pre, models[best_k], fam.name, EMPHASIS)
        labels_all = segmenter.predict(d.df)
        prof = segment_outcomes(d.df.customer_id, labels_all)

        stem = f"d03_{fam.name}"
        sweep.to_csv(DS_REPORTS_DIR / f"{stem}_sweep.csv", index=False)
        prof.to_csv(DS_REPORTS_DIR / f"{stem}_segments.csv")
        plot_sweep(sweep, fam, best_k, DS_REPORTS_DIR / f"{stem}_sweep.png")
        for suffix in ("_sweep.csv", "_segments.csv", "_sweep.png"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / f"{stem}{suffix}"))
        mlflow.log_params({"reference_k": best_k, "reference_admissible": bool(best.admissible)})
        mlflow.log_metrics({**{f"reference_{c}": float(best[c]) for c in
                               ("silhouette_gower", "silhouette_native", "stability_ari_mean", "min_share")}, **sens})
        with tracking.staged_code_paths() as code_paths:
            mlflow.pyfunc.log_model(name=f"segmenter_{fam.name}", python_model=SegmenterPyfunc(segmenter),
                                    code_paths=code_paths, input_example=d.df.head(5))

        candidate_id = f"{fam.name}-k{best_k}"
        metrics = {**{c: float(best[c]) for c in ("silhouette_gower", "silhouette_native", "calinski_harabasz",
                                                   "davies_bouldin", "stability_ari_mean", "min_share", "max_share")},
                   **sens}
        save_candidate(candidate_id, fam.name, best_k, FEATURE_SPACE, parent.info.run_id, metrics,
                       d.df.customer_id, labels_all)
        print(prof.to_string())
        print(f"  saved candidate {candidate_id}; MLflow run {parent.info.run_id}")
        return {"family": fam.name, "candidate_id": candidate_id, "k": best_k, **best.to_dict(), **sens}


def main(families: list[str] | None = None) -> None:
    names = families or list(FAMILIES)
    unknown = set(names) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"Unknown families {sorted(unknown)}; choose from {list(FAMILIES)}")
    DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Preparing curated training data")
    d = Data()
    print(f"  {len(d.train_df):,} customers; encoded {d.encoded[EMPHASIS].shape[1]} columns; "
          f"k-prototypes fit sample {len(d.fit_sample):,}")
    tracking.setup()
    summary = pd.DataFrame([run_family(d, FAMILIES[n]) for n in names])
    print("\n" + summary[["candidate_id", "silhouette_gower", "stability_ari_mean", "min_share"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
