"""Statistics helpers — the 'full rigor' layer.

Everything the dashboard reports about an edge comes through here so the
confidence intervals and standard errors are computed consistently.
"""

from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054  # two-sided 95% normal critical value


@dataclass
class EdgeStats:
    n: int
    mean: float          # mean result per round, in units (negative => house wins)
    std: float
    sem: float           # standard error of the mean
    ci_low: float
    ci_high: float

    @property
    def house_edge_pct(self) -> float:
        """House edge as a percentage of the base bet (positive = house wins)."""
        return -self.mean * 100.0

    @property
    def ci_pct(self):
        """95% CI on the house edge, in percent (low, high)."""
        return (-self.ci_high * 100.0, -self.ci_low * 100.0)


def summarize(results: np.ndarray) -> EdgeStats:
    """Summarize an array of per-round results (in base-bet units)."""
    results = np.asarray(results, dtype=float)
    n = results.size
    mean = float(results.mean())
    std = float(results.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    return EdgeStats(
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        ci_low=mean - Z95 * sem,
        ci_high=mean + Z95 * sem,
    )


def running_mean(results: np.ndarray) -> np.ndarray:
    """Cumulative average after each round — the convergence curve."""
    results = np.asarray(results, dtype=float)
    return np.cumsum(results) / np.arange(1, results.size + 1)


def running_sem(results: np.ndarray) -> np.ndarray:
    """Standard error of the running mean at each round (for CI bands)."""
    results = np.asarray(results, dtype=float)
    n = np.arange(1, results.size + 1)
    csum = np.cumsum(results)
    csum2 = np.cumsum(results ** 2)
    mean = csum / n
    # population->sample variance; guard the first sample
    var = np.maximum(csum2 / n - mean ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sample_var = var * n / np.maximum(n - 1, 1)
        sem = np.sqrt(sample_var / n)
    sem[0] = 0.0
    return sem
