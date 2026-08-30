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

    # Paired t-test
    try:
        t_stat, p_val_ttest = stats.ttest_rel(a, b)
        t_stat = float(t_stat)
        p_val_ttest = float(p_val_ttest)
    except Exception:
        t_stat, p_val_ttest = 0.0, 1.0

    # Wilcoxon signed-rank test
    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(a, b)
        w_stat = float(w_stat)
        p_val_wilcoxon = float(p_val_wilcoxon)
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
        "t_statistic": round(t_stat, 4),
        "p_value_ttest": round(p_val_ttest, 5),
        "ttest_significant_05": p_val_ttest < 0.05,
        "wilcoxon_statistic": round(w_stat, 4),
        "p_value_wilcoxon": round(p_val_wilcoxon, 5),
        "wilcoxon_significant_05": p_val_wilcoxon < 0.05,
    }
