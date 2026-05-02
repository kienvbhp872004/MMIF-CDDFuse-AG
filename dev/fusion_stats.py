"""
Statistical comparison: variant vs baseline (paired per-image).

- Wilcoxon signed-rank (paired), alternative='greater' (variant > baseline after sign-flip
  for lower-better metrics).
- Cliff's delta (effect size).
- Holm-Bonferroni correction across K metrics.
- Per-modality + pooled (ALL).

Usage:
    python dev/fusion_stats.py --variant FuseRule-Gated
                               [--baseline CDDFuse]
                               [--alpha 0.05]
"""
import argparse
import csv
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results_v2"

LOWER_BETTER = {"NABF", "CE", "RMSE", "QCV"}
METRICS = [
    "EN", "VAR", "AG", "SF", "EI", "MI", "NCIE", "MI_mutual", "NABF", "FMI",
    "CE", "SSIM", "PSNR", "RMSE", "QG", "QM", "QC", "QS", "QCB", "QCV",
    "QY", "QMI", "QSF", "QNCIE", "QTE",
]
MODALS = ["CT", "PET", "SPECT"]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(X > Y) - P(X < Y). Range [-1, 1]."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float((gt - lt) / n)


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down. Input p, output p_adj."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    p_adj_sorted = np.empty(n)
    prev = 0.0
    for i, pi in enumerate(p_sorted):
        adj = max(prev, min(1.0, pi * (n - i)))
        p_adj_sorted[i] = adj
        prev = adj
    p_adj = np.empty(n)
    p_adj[order] = p_adj_sorted
    return p_adj


def load_perimage(model_dir: Path, model_name: str, modal: str) -> pd.DataFrame:
    """Load one perimage CSV for a model + modal."""
    fp = model_dir / "perimage" / f"{model_name}_{modal}_perimage.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing per-image CSV: {fp}. Run with --save_perimage first.")
    return pd.read_csv(fp)


def run_tests(df_pair: pd.DataFrame, suffix_base: str, suffix_var: str) -> List[dict]:
    """Run Wilcoxon + Cliff's δ for each metric on a paired dataframe."""
    rows = []
    for m in METRICS:
        col_b = f"{m}{suffix_base}"
        col_v = f"{m}{suffix_var}"
        if col_b not in df_pair.columns or col_v not in df_pair.columns:
            continue
        base = df_pair[col_b].dropna().to_numpy()
        var = df_pair[col_v].dropna().to_numpy()
        # paired: use same indices that have both non-null
        joint = df_pair[[col_b, col_v]].dropna()
        if len(joint) < 5:
            continue
        b = joint[col_b].to_numpy()
        v = joint[col_v].to_numpy()
        diff = v - b
        if m in LOWER_BETTER:
            diff = -diff  # variant better when lower
        # If all diffs are zero -> p=1
        if np.all(diff == 0):
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = wilcoxon(diff, alternative="greater", zero_method="wilcox")
            except Exception:
                stat, pval = 0.0, 1.0
        rows.append({
            "metric": m,
            "n": len(joint),
            "mean_base": float(np.mean(b)),
            "mean_var": float(np.mean(v)),
            "delta": float(np.mean(v) - np.mean(b)),
            "cliffs_delta": cliffs_delta(v, b) * (-1 if m in LOWER_BETTER else 1),
            "wilcoxon_stat": float(stat),
            "p_value": float(pval),
        })
    return rows


def verdict_for(p_adj: float, alpha: float, p: float) -> str:
    if p_adj < alpha:
        return "SIG"
    if p < alpha:
        return "MARGINAL"  # passes raw but not Holm-corrected
    return "NS"


def effect_size_label(d: float) -> str:
    a = abs(d)
    if a < 0.147: return "trivial"
    if a < 0.33:  return "small"
    if a < 0.474: return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="e.g. FuseRule-Gated")
    ap.add_argument("--baseline", default="CDDFuse")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    var_name = f"CDDFuse-{args.variant}" if not args.variant.startswith("CDDFuse") else args.variant
    base_name = args.baseline

    base_dir = RESULTS / base_name
    var_dir = RESULTS / var_name
    if not base_dir.exists() or not var_dir.exists():
        raise FileNotFoundError(f"{base_dir} or {var_dir} missing")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / "_stats" / f"{run_id}_{args.variant}_vs_{base_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stats] output -> {out_dir}")

    all_rows = []
    pooled_cols: Dict[str, list] = {f"{m}_base": [] for m in METRICS}
    pooled_cols.update({f"{m}_var": [] for m in METRICS})

    for modal in MODALS:
        df_b = load_perimage(base_dir, base_name, modal)
        df_v = load_perimage(var_dir, var_name, modal)
        df = df_b.merge(df_v, on="image", suffixes=("_base", "_var"))
        n = len(df)
        if n == 0:
            print(f"[stats] WARN: 0 paired images for {modal}")
            continue
        print(f"[stats] {modal}: {n} paired images")
        rows = run_tests(df, "_base", "_var")
        for r in rows:
            r["modal"] = modal
        all_rows.extend(rows)

        for m in METRICS:
            if f"{m}_base" in df.columns:
                pooled_cols[f"{m}_base"].extend(df[f"{m}_base"].tolist())
                pooled_cols[f"{m}_var"].extend(df[f"{m}_var"].tolist())

    # Pooled across modalities
    pooled_df = pd.DataFrame(pooled_cols)
    pooled_rows = run_tests(pooled_df, "_base", "_var")
    for r in pooled_rows:
        r["modal"] = "ALL"
    all_rows.extend(pooled_rows)

    # Holm correction PER MODAL group (so K = 22 within each modal)
    by_modal: Dict[str, list] = {}
    for r in all_rows:
        by_modal.setdefault(r["modal"], []).append(r)
    for modal, rows in by_modal.items():
        ps = np.array([r["p_value"] for r in rows])
        p_adj = holm_correction(ps)
        for r, pa in zip(rows, p_adj):
            r["p_adj_holm"] = float(pa)
            r["verdict"] = verdict_for(pa, args.alpha, r["p_value"])
            r["effect"] = effect_size_label(r["cliffs_delta"])

    # Write CSV
    fieldnames = ["metric", "modal", "n", "mean_base", "mean_var", "delta",
                  "cliffs_delta", "effect", "wilcoxon_stat",
                  "p_value", "p_adj_holm", "verdict"]
    csv_path = out_dir / f"significance_{args.variant}_vs_{base_name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"[stats] wrote {csv_path}")

    # Build report
    n_sig_all = sum(1 for r in all_rows if r["modal"] == "ALL" and r["verdict"] == "SIG")
    n_marg_all = sum(1 for r in all_rows if r["modal"] == "ALL" and r["verdict"] == "MARGINAL")
    n_ns_all = sum(1 for r in all_rows if r["modal"] == "ALL" and r["verdict"] == "NS")
    n_total_all = sum(1 for r in all_rows if r["modal"] == "ALL")

    # Largest improvements
    all_only = [r for r in all_rows if r["modal"] == "ALL"]
    all_only_sorted = sorted(all_only, key=lambda r: r["cliffs_delta"], reverse=True)
    top5_pos = all_only_sorted[:5]
    top5_neg = all_only_sorted[-5:][::-1]

    if n_sig_all >= 5:
        verdict_overall = "CONFIRM_IMPROVEMENT"
    elif n_sig_all + n_marg_all == 0:
        verdict_overall = "INSUFFICIENT_EVIDENCE"
    else:
        verdict_overall = "MIXED"

    md = []
    md.append(f"# Statistical comparison: `CDDFuse-{args.variant}` vs `{base_name}` baseline\n")
    md.append(f"**Run ID**: {run_id}")
    md.append(f"**Alpha**: {args.alpha}, correction: Holm-Bonferroni (per modal)")
    md.append(f"**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)")
    md.append(f"**Output**: `{csv_path.relative_to(REPO)}`\n")
    md.append("## Bottom line\n")
    md.append(f"- Verdict: **{verdict_overall}**")
    md.append(f"- Pooled (ALL): {n_sig_all}/{n_total_all} metrics SIG, {n_marg_all} marginal, {n_ns_all} NS")
    md.append("\n## Top 5 metrics with largest improvement (pooled)\n")
    md.append("| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for r in top5_pos:
        md.append(f"| {r['metric']} | {r['delta']:+.4f} | {r['cliffs_delta']:+.3f} | {r['effect']} | {r['p_value']:.4f} | {r['p_adj_holm']:.4f} | **{r['verdict']}** |")

    md.append("\n## Top 5 metrics with largest regression (pooled)\n")
    md.append("| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for r in top5_neg:
        md.append(f"| {r['metric']} | {r['delta']:+.4f} | {r['cliffs_delta']:+.3f} | {r['effect']} | {r['p_value']:.4f} | {r['p_adj_holm']:.4f} | {r['verdict']} |")

    md.append("\n## Per-modality summary\n")
    md.append("| modal | n_SIG | n_MARG | n_NS | total |")
    md.append("|---|---|---|---|---|")
    for modal in MODALS + ["ALL"]:
        rows = [r for r in all_rows if r["modal"] == modal]
        if not rows:
            continue
        sig = sum(1 for r in rows if r["verdict"] == "SIG")
        mg = sum(1 for r in rows if r["verdict"] == "MARGINAL")
        ns = sum(1 for r in rows if r["verdict"] == "NS")
        md.append(f"| {modal} | {sig} | {mg} | {ns} | {len(rows)} |")

    md.append("\n## Caveats")
    md.append(f"- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.")
    md.append(f"- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.")
    md.append(f"- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.\n")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[stats] wrote {report_path}")

    # JSON summary for programmatic consumption
    summary = {
        "run_id": run_id,
        "variant": var_name,
        "baseline": base_name,
        "alpha": args.alpha,
        "verdict_overall": verdict_overall,
        "pooled": {"n_sig": n_sig_all, "n_marginal": n_marg_all, "n_ns": n_ns_all, "n_total": n_total_all},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stats] DONE. Verdict: {verdict_overall}")


if __name__ == "__main__":
    main()
