"""Statistics for the free-form results: confidence intervals, tests, and multiple-comparison
correction."""
import math

from scipy import stats as sps


# --------------------------------------------------------- the paper's tests
def wilson_ci(successes, n, conf=0.95):
    """Wilson score interval for a proportion. Well behaved near 0 and 1, unlike the
    normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(s1, n1, s2, n2):
    """Two-proportion z-test (SycEval's test for preemptive vs in-context). Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = s1 / n1, s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - sps.norm.cdf(abs(z)))


def chi_square(table):
    """[PAPER] Chi-square test of independence over a contingency table
    (list of rows). Returns (chi2, p, dof)."""
    rows = [r for r in table if sum(r) > 0]
    if len(rows) < 2 or len(rows[0]) < 2:
        return float("nan"), float("nan"), 0
    chi2, p, dof, _ = sps.chi2_contingency(rows)
    return chi2, p, dof


def binomial_test(successes, n, p0=0.5):
    """[PAPER] Persistence is tested against a 50% baseline with a binomial test."""
    if n == 0:
        return float("nan")
    return sps.binomtest(successes, n, p0).pvalue


# ----------------------------------------------------- this repo's item-level
def paired_item_test(pairs):
    """Paired Wilcoxon test over (rate_a, rate_b), one pair per question.
    Returns (mean_diff, p, n_items, n_down, n_up, n_tie)."""
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), 0, 0, 0, 0
    down = sum(1 for d in diffs if d < 0)
    up = sum(1 for d in diffs if d > 0)
    tie = sum(1 for d in diffs if d == 0)
    mean = sum(diffs) / n
    if up + down == 0:
        return mean, 1.0, n, down, up, tie
    p = sps.wilcoxon(a, b, zero_method="zsplit").pvalue
    return mean, p, n, down, up, tie


def holm(pvalues):
    """Holm correction for several p-values. Takes {label: p}, returns {label: adjusted p}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(running, min(1.0, p * (m - i)))
        out[k] = adj
        running = adj
    return out


def beta_judge_accuracy(matches, mismatches):
    """Judge accuracy from a human check, as a Beta(matches + 1, mismatches + 1) distribution.
    Returns (mean, lower 95%, upper 95%)."""
    a, b = matches + 1, mismatches + 1
    dist = sps.beta(a, b)
    return dist.mean(), dist.ppf(0.025), dist.ppf(0.975)


def fmt_p(p):
    if p != p:
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def rate_row(successes, n):
    """'43.5% [40.1, 47.0]  n=1234' -- the paper's reporting format."""
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(successes, n)
    return f"{successes / n:.1%} [{lo:.1%}, {hi:.1%}]  n={n}"
