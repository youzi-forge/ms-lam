from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_int_pair(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected two integers like '2,6'")
    return int(parts[0]), int(parts[1])


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5: exploratory phenotyping (PCA + k-means + stability).")
    ap.add_argument(
        "--features-csv",
        type=str,
        default="results/tables/features_v1.csv",
        help="Input features table (from scripts/12_phase5_export_features.py).",
    )
    ap.add_argument(
        "--feature-set",
        type=str,
        default="mode_a",
        choices=["mode_a", "mode_a_pheno", "mode_a_eval", "mode_b"],
        help="Which feature subset to cluster on (mode_a=Phase2+Phase4; mode_b adds Phase3 sensitivity if present).",
    )
    ap.add_argument(
        "--missing",
        type=str,
        default="drop",
        choices=["drop", "median"],
        help="How to handle missing values in selected features (drop rows, or median-impute).",
    )
    ap.add_argument("--k", type=int, default=0, help="Number of clusters. If 0, select by silhouette over --k-range.")
    ap.add_argument("--k-range", type=str, default="2,6", help="Inclusive k-range for silhouette selection, like '2,6'.")
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="When selecting k automatically, skip solutions with clusters smaller than this size (useful to avoid singleton outlier clusters).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=50, help="Number of random seeds for stability estimation.")
    ap.add_argument("--n-init", type=int, default=50, help="KMeans n_init for each run.")
    ap.add_argument(
        "--scaler",
        type=str,
        default="robust",
        choices=["robust", "standard"],
        help="Feature scaling before PCA/clustering.",
    )
    ap.add_argument(
        "--annotate",
        type=str,
        default="all",
        choices=["none", "all", "qc", "extremes"],
        help="Which points to annotate with patient_id in the latent-space figure.",
    )
    ap.add_argument(
        "--qc-encoding",
        type=str,
        default="marker",
        choices=["marker", "edge", "none"],
        help='How to visualize qc_needs_review on the latent-space plot. "marker" uses shape; "edge" uses edge color; "none" disables QC styling.',
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--out-assignments",
        type=str,
        default="results/tables/phenotype_assignments.csv",
        help="Output CSV with cluster assignments and PCA coordinates.",
    )
    ap.add_argument(
        "--out-profiles",
        type=str,
        default="results/tables/phase5_cluster_profiles.csv",
        help="Output CSV summarizing per-cluster feature profiles (medians).",
    )
    ap.add_argument(
        "--out-fig-latent",
        type=str,
        default="results/figures/phase5_latent_space_pca.png",
        help="Output PCA latent-space figure.",
    )
    ap.add_argument(
        "--out-fig-stability",
        type=str,
        default="results/figures/phase5_coassignment_heatmap.png",
        help="Output stability co-assignment heatmap.",
    )
    ap.add_argument(
        "--out-k-metrics",
        type=str,
        default="results/tables/phase5_k_selection.csv",
        help="Output CSV with silhouette scores by k (for transparency).",
    )
    args = ap.parse_args()

    repo = _repo_root()
    in_csv = (repo / args.features_csv).resolve()
    out_assign = (repo / args.out_assignments).resolve()
    out_profiles = (repo / args.out_profiles).resolve()
    out_latent = (repo / args.out_fig_latent).resolve()
    out_stab = (repo / args.out_fig_stability).resolve()
    out_k = (repo / args.out_k_metrics).resolve()

    # Matplotlib cache local+writable (avoids slow font-cache rebuilds in some environments).
    mpl_cache = repo / "results" / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    # Silence occasional loky warnings about physical-core detection on macOS.
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

    import numpy as np
    import pandas as pd

    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import RobustScaler, StandardScaler
    except Exception as e:
        raise SystemExit(
            "scikit-learn is required for Phase 5. Install deps via:\n\n"
            "  pip3 install -r requirements.txt\n\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    df = pd.read_csv(in_csv)
    if "patient_id" not in df.columns:
        raise SystemExit("features table must include patient_id")

    # Define feature subsets (Mode A uses only monitoring + uncertainty; excludes change-GT dependent metrics by default).
    feature_sets: dict[str, list[str]] = {
        "mode_a": [
            "log1p_lesion_vol_t0_mm3",
            "log1p_lesion_vol_t1_mm3",
            "delta_lesion_vol_mm3",
            "log1p_new_cons_vol_mm3",
            "intensity_diff_mean",
            "intensity_diff_p95",
            "unc_ens_var_mean_lesion_t1",
            "unc_ens_var_p95_brain_t1",
        ],
        # Phenotyping-oriented variant: exclude the global tail-uncertainty (often a QC/outlier axis).
        # Keep lesion-mean uncertainty as a quality-aware feature, but avoid letting brain-wide p95 dominate clustering.
        "mode_a_pheno": [
            "log1p_lesion_vol_t0_mm3",
            "log1p_lesion_vol_t1_mm3",
            "delta_lesion_vol_mm3",
            "log1p_new_cons_vol_mm3",
            "intensity_diff_mean",
            "intensity_diff_p95",
            "unc_ens_var_mean_lesion_t1",
        ],
        "mode_a_eval": [
            "log1p_lesion_vol_t0_mm3",
            "log1p_lesion_vol_t1_mm3",
            "delta_lesion_vol_mm3",
            "log1p_new_cons_vol_mm3",
            "intensity_diff_mean",
            "intensity_diff_p95",
            "unc_ens_var_mean_lesion_t1",
            "unc_ens_var_p95_brain_t1",
            # Evaluation metrics (dataset-dependent; to change-GT)
            "dice_new_cons",
            "dice_chg_sym_cons",
        ],
        "mode_b": [
            "log1p_lesion_vol_t0_mm3",
            "log1p_lesion_vol_t1_mm3",
            "delta_lesion_vol_mm3",
            "log1p_new_cons_vol_mm3",
            "intensity_diff_mean",
            "intensity_diff_p95",
            "unc_ens_var_mean_lesion_t1",
            "unc_ens_var_p95_brain_t1",
            # Robustness sensitivity (if available; typically from Phase 3×4 merge)
            "log1p_phase3_deltav_abs_p95",
            "phase3_dice_sym_drop_p05",
        ],
    }
    features = feature_sets[str(args.feature_set)]
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required feature columns for {args.feature_set}: {missing_cols}")

    meta_cols = ["patient_id"]
    for c in ["qc_needs_review", "qc_change_not_confident"]:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)

    sub = df[meta_cols + features].copy()
    # Coerce QC flags to bool if present.
    for c in ["qc_needs_review", "qc_change_not_confident"]:
        if c in sub.columns:
            sub[c] = sub[c].map(_to_bool)

    # Handle missing values.
    mask_complete = sub[features].notna().all(axis=1)
    if not bool(mask_complete.all()):
        n_missing = int((~mask_complete).sum())
        if args.missing == "drop":
            sub = sub[mask_complete].copy()
            if len(sub) < 3:
                raise SystemExit("Too few rows after dropping missing feature rows.")
            print(f"[phase5] Dropped {n_missing} rows with missing features (missing=drop).")
        else:
            med = sub[features].median(numeric_only=True)
            sub[features] = sub[features].fillna(med)
            print(f"[phase5] Median-imputed {n_missing} rows with missing features (missing=median).")

    patient_ids = sub["patient_id"].astype(str).tolist()
    X_raw = sub[features].astype(float).to_numpy()

    scaler = RobustScaler() if args.scaler == "robust" else StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=int(args.seed))
    X_pca = pca.fit_transform(X_scaled)

    # Select k by silhouette on PCA space (or use provided k).
    k_low, k_high = _parse_int_pair(args.k_range)
    k_low = max(2, int(k_low))
    k_high = max(k_low, int(k_high))
    k_candidates = [k for k in range(k_low, min(k_high, len(sub) - 1) + 1)]
    if not k_candidates:
        raise SystemExit("No valid k candidates (check --k-range vs number of patients).")

    k_metrics_rows: list[dict[str, float | int]] = []
    if int(args.k) > 0:
        k_best = int(args.k)
    else:
        best = None
        for k in k_candidates:
            km = KMeans(n_clusters=int(k), n_init=int(args.n_init), random_state=int(args.seed))
            labels = km.fit_predict(X_pca)
            # Optional constraint to avoid singleton/outlier-only clusters when auto-selecting k.
            min_size = int(np.bincount(labels).min()) if len(labels) else 0
            if int(args.min_cluster_size) > 1 and min_size < int(args.min_cluster_size):
                continue
            score = float(silhouette_score(X_pca, labels))
            k_metrics_rows.append({"k": int(k), "silhouette": float(score)})
            if best is None or score > best[0] or (score == best[0] and k < best[1]):
                best = (score, int(k))
        assert best is not None
        k_best = int(best[1])

    if k_metrics_rows:
        _ensure_parent(out_k)
        pd.DataFrame(k_metrics_rows).sort_values("k").to_csv(out_k, index=False)

    km_final = KMeans(n_clusters=int(k_best), n_init=int(args.n_init), random_state=int(args.seed))
    labels_final = km_final.fit_predict(X_pca)
    if int(args.k) > 0 and int(args.min_cluster_size) > 1:
        min_size_final = int(np.bincount(labels_final).min())
        if min_size_final < int(args.min_cluster_size):
            print(
                f"[phase5] Warning: min cluster size={min_size_final} < --min-cluster-size={int(args.min_cluster_size)} (k fixed by user)."
            )

    # Stability: co-assignment probability across repeated kmeans seeds.
    n = len(sub)
    co = np.zeros((n, n), dtype=np.float32)
    seeds = [int(args.seed) + i for i in range(int(args.n_seeds))]
    for s in seeds:
        km = KMeans(n_clusters=int(k_best), n_init=max(10, int(args.n_init) // 2), random_state=int(s))
        lab = km.fit_predict(X_pca)
        for c in range(int(k_best)):
            idx = np.where(lab == c)[0]
            if len(idx) == 0:
                continue
            co[np.ix_(idx, idx)] += 1.0
    co /= float(len(seeds))

    # Per-sample stability: mean co-assignment with its reference-cluster mates.
    stability = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mates = np.where(labels_final == labels_final[i])[0]
        if len(mates) <= 1:
            stability[i] = 1.0
        else:
            mates_wo = mates[mates != i]
            stability[i] = float(co[i, mates_wo].mean())

    # Save assignments (with small set of interpretable raw columns for context).
    out = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "cluster_id": labels_final.astype(int),
            "pca1": X_pca[:, 0].astype(float),
            "pca2": X_pca[:, 1].astype(float) if X_pca.shape[1] > 1 else 0.0,
            "stability_in_cluster": stability.astype(float),
        }
    )
    for c in ["qc_needs_review", "qc_change_not_confident"]:
        if c in sub.columns:
            out[c] = sub[c].astype(bool).to_list()

    # Add a small set of raw metrics for interpretability (not used for clustering).
    for c in ["lesion_vol_t0_mm3", "lesion_vol_t1_mm3", "delta_lesion_vol_mm3", "new_cons_vol_mm3", "unc_ens_var_mean_lesion_t1"]:
        if c in df.columns:
            out[c] = df.set_index("patient_id").loc[out["patient_id"], c].astype(float).to_numpy()

    _ensure_parent(out_assign)
    out.to_csv(out_assign, index=False)

    # Cluster profiles (medians) for interpretability.
    prof_cols = [c for c in features if c in sub.columns]
    prof = sub[["patient_id"] + prof_cols].copy()
    prof["cluster_id"] = labels_final.astype(int)
    profiles = prof.groupby("cluster_id", as_index=False).median(numeric_only=True)
    sizes = prof.groupby("cluster_id")["patient_id"].count().rename("n_patients").reset_index()
    profiles = pd.merge(sizes, profiles, on="cluster_id", how="inner")
    # QC rates per cluster (if available)
    if "qc_needs_review" in sub.columns:
        qc_rate = (
            pd.DataFrame({"cluster_id": labels_final.astype(int), "qc_needs_review": sub["qc_needs_review"].astype(bool).to_list()})
            .groupby("cluster_id", as_index=False)["qc_needs_review"]
            .mean()
            .rename(columns={"qc_needs_review": "qc_needs_review_rate"})
        )
        profiles = pd.merge(profiles, qc_rate, on="cluster_id", how="left")

    _ensure_parent(out_profiles)
    profiles.to_csv(out_profiles, index=False)

    # Figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Latent space
    fig = plt.figure(figsize=(8, 6), layout="constrained")
    ax = fig.subplots()
    clusters = sorted(set(labels_final.astype(int).tolist()))
    cmap = plt.get_cmap("tab10")

    qc = sub["qc_needs_review"].astype(bool).to_numpy() if "qc_needs_review" in sub.columns else np.zeros(n, dtype=bool)
    for c in clusters:
        idx = np.where(labels_final.astype(int) == c)[0]
        color = cmap(int(c) % 10)
        y_vals = X_pca[idx, 1] if X_pca.shape[1] > 1 else np.zeros(len(idx), dtype=float)

        if args.qc_encoding == "marker" and qc.any():
            idx_ok = idx[~qc[idx]]
            idx_qc = idx[qc[idx]]
            labeled = False
            if len(idx_ok) > 0:
                ax.scatter(
                    X_pca[idx_ok, 0],
                    y_vals[~qc[idx]],
                    s=70,
                    c=[color],
                    marker="o",
                    label=f"cluster {c}",
                    edgecolors="black",
                    linewidths=1.2,
                    alpha=0.9,
                )
                labeled = True
            if len(idx_qc) > 0:
                ax.scatter(
                    X_pca[idx_qc, 0],
                    y_vals[qc[idx]],
                    s=90,
                    c=[color],
                    marker="^",
                    label=(f"cluster {c}" if not labeled else None),
                    edgecolors="black",
                    linewidths=1.2,
                    alpha=0.9,
                )
        else:
            if args.qc_encoding == "edge" and qc.any():
                edgecolors = ["#d62728" if qc[i] else "black" for i in idx]  # red highlights QC
            else:
                edgecolors = "black"
            ax.scatter(
                X_pca[idx, 0],
                y_vals,
                s=70,
                c=[color],
                label=f"cluster {c}",
                edgecolors=edgecolors,
                linewidths=1.2,
                alpha=0.9,
            )

    ax.set_title(f"Phase 5: PCA latent space + k-means (k={k_best}, features={args.feature_set})", fontsize=12)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    # Two legends: clusters (color) and QC (shape/edge), to avoid visual ambiguity.
    cluster_leg = ax.legend(loc="best", fontsize=9, title="Cluster")
    if args.qc_encoding == "marker" and qc.any():
        from matplotlib.lines import Line2D

        qc_handles = [
            Line2D([0], [0], marker="o", color="w", label="qc: ok", markerfacecolor="lightgray", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="^", color="w", label="qc: needs_review", markerfacecolor="lightgray", markeredgecolor="black", markersize=8),
        ]
        ax.add_artist(cluster_leg)
        ax.legend(handles=qc_handles, loc="lower right", fontsize=9, title="QC")
    elif args.qc_encoding == "edge" and qc.any():
        from matplotlib.lines import Line2D

        qc_handles = [
            Line2D([0], [0], marker="o", color="w", label="qc: ok", markerfacecolor="lightgray", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="qc: needs_review", markerfacecolor="lightgray", markeredgecolor="#d62728", markersize=8),
        ]
        ax.add_artist(cluster_leg)
        ax.legend(handles=qc_handles, loc="lower right", fontsize=9, title="QC")

    def annotate_mask() -> np.ndarray:
        if args.annotate == "none":
            return np.zeros(n, dtype=bool)
        if args.annotate == "all":
            return np.ones(n, dtype=bool)
        if args.annotate == "qc":
            return qc
        # extremes: top-3 by |pca1|+|pca2|
        score = np.abs(X_pca[:, 0]) + np.abs(X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0)
        k = min(3, n)
        idx = np.argsort(score)[-k:]
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return m

    m = annotate_mask()
    offsets = [(4, 2), (4, -8), (-44, 2), (-44, -8)]
    for j, i in enumerate(np.where(m)[0]):
        dx, dy = offsets[j % len(offsets)]
        ax.annotate(patient_ids[int(i)], (X_pca[int(i), 0], X_pca[int(i), 1] if X_pca.shape[1] > 1 else 0.0), fontsize=8, xytext=(dx, dy), textcoords="offset points")

    _ensure_parent(out_latent)
    fig.savefig(out_latent, dpi=int(args.dpi))
    plt.close(fig)

    # Co-assignment heatmap (sorted by reference cluster then PCA1)
    order = sorted(range(n), key=lambda i: (int(labels_final[i]), float(X_pca[i, 0])))
    co_s = co[np.ix_(order, order)]
    labels_s = [patient_ids[i] for i in order]

    fig = plt.figure(figsize=(10, 9), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(co_s, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"Phase 5: co-assignment stability (n_runs={len(seeds)}, k={k_best})", fontsize=12)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_s, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_s, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(same cluster)")

    _ensure_parent(out_stab)
    fig.savefig(out_stab, dpi=int(args.dpi))
    plt.close(fig)

    print(f"Wrote: {out_assign.relative_to(repo)}")
    print(f"Wrote: {out_profiles.relative_to(repo)}")
    print(f"Wrote: {out_latent.relative_to(repo)}")
    print(f"Wrote: {out_stab.relative_to(repo)}")
    if k_metrics_rows:
        print(f"Wrote: {out_k.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
