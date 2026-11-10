import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple

def run_paired_tests(array_a: List[float], array_b: List[float], metric_name: str) -> Dict[str, Any]:
    """
    Computes paired t-test and Wilcoxon signed-rank test between two paired metric arrays.
    Returns statistical test summary dictionary.
    """
    a = np.array(array_a, dtype=np.float64)
    b = np.array(array_b, dtype=np.float64)

    if len(a) != len(b) or len(a) < 2:
        return {"metric": metric_name, "error": "Insufficient paired samples"}

    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

    # A zero-variance difference is not a usable t-test statistic.
    if np.allclose(diff, 0.0):
        t_stat, p_val_ttest = 0.0, 1.0
    elif std_diff <= 1e-12:
        t_stat, p_val_ttest = None, None
    else:
        try:
            t_stat, p_val_ttest = stats.ttest_rel(a, b)
            t_stat = float(t_stat) if np.isfinite(t_stat) else None
            p_val_ttest = float(p_val_ttest) if np.isfinite(p_val_ttest) else None
        except Exception:
            t_stat, p_val_ttest = None, None

    # Wilcoxon signed-rank test
    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(a, b)
        w_stat = float(w_stat) if np.isfinite(w_stat) else None
        p_val_wilcoxon = float(p_val_wilcoxon) if np.isfinite(p_val_wilcoxon) else None
    except Exception:
        w_stat, p_val_wilcoxon = 0.0, 1.0

    # Cohen's d effect size
    cohen_d = float(mean_diff / std_diff) if std_diff > 1e-6 else 0.0

    return {
        "metric": metric_name,
        "n_samples": len(a),
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "cohen_d": round(cohen_d, 4),
        "t_statistic": round(t_stat, 4) if t_stat is not None else None,
        "p_value_ttest": round(p_val_ttest, 5) if p_val_ttest is not None else None,
        "ttest_significant_05": p_val_ttest < 0.05 if p_val_ttest is not None else None,
        "wilcoxon_statistic": round(w_stat, 4) if w_stat is not None else None,
        "p_value_wilcoxon": round(p_val_wilcoxon, 5) if p_val_wilcoxon is not None else None,
        "wilcoxon_significant_05": p_val_wilcoxon < 0.05 if p_val_wilcoxon is not None else None,
    }
