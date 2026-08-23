"""The promotion gate. Pure: metrics in, decision out. No I/O, no MLflow.

Non-promotion is a successful decision, never an error.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    observed: float | None
    threshold: float | None
    reason: str


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    checks: tuple[CheckResult, ...]
    delta_observed: float | None
    ci_low: float | None
    ci_high: float | None
    reason: str


def _get_ci_range(
    candidate_metrics: dict, champion_metrics: dict, n_boot: int, seed: int
) -> tuple[float, float]:
    """Paired bootstrap 95% CI on the MAE delta (champion - candidate).

    Each round draws one shared index sample for BOTH models — the
    pairing cancels sample-difficulty variance, leaving model skill.
    """
    c_abs_errs = np.asarray(candidate_metrics["abs_errors"])
    ch_abs_errs = np.asarray(champion_metrics["abs_errors"])

    n = len(c_abs_errs)

    sample_gen = np.random.default_rng(seed)
    idxs = sample_gen.integers(0, n, size=(n_boot, n))

    c_avg = c_abs_errs[idxs].mean(axis=1)
    ch_avg = ch_abs_errs[idxs].mean(axis=1)

    deltas = ch_avg - c_avg  # positive = candidate better
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _check_champion_exists(champion_metrics: dict | None) -> CheckResult:
    """Records which path the gate took. Cold start is a state, not a failure."""
    champion_is_none = champion_metrics is None

    return CheckResult(
        name="cold_start",
        passed=champion_is_none,
        observed=None,
        threshold=None,
        reason="no champion - candidate becomes first champion"
        if champion_is_none
        else "champion exists - proceed to comparison",
    )


def _check_sample_floor(candidate_metrics: dict, gate_config: dict) -> CheckResult:
    num_samples = len(candidate_metrics["abs_errors"])
    min_samples = gate_config["min_samples"]

    passed = num_samples >= min_samples

    return CheckResult(
        name="sample_floor",
        passed=passed,
        observed=num_samples,
        threshold=min_samples,
        reason="sufficient held-out samples"
        if passed
        else f"insufficient evidence: {num_samples} below floor of {min_samples}",
    )


def gate_decision(
    candidate_metrics: dict,
    champion_metrics: dict | None,
    gate_config: dict,
) -> GateDecision:
    checks: list[CheckResult] = []

    passed_floor = _check_sample_floor(candidate_metrics, gate_config)
    checks.append(passed_floor)

    if not passed_floor.passed:
        return GateDecision(
            promote=False,
            checks=tuple(checks),
            delta_observed=None,
            ci_low=None,
            ci_high=None,
            reason=passed_floor.reason,
        )

    cold_start = _check_champion_exists(champion_metrics)
    checks.append(cold_start)

    if cold_start.passed:
        return GateDecision(
            promote=True,
            checks=tuple(checks),
            delta_observed=None,
            ci_low=None,
            ci_high=None,
            reason=cold_start.reason,
        )

    delta_observed = float(champion_metrics["mae"] - candidate_metrics["mae"])
    ci_low, ci_high = _get_ci_range(
        candidate_metrics,
        champion_metrics,
        n_boot=gate_config["n_boot"],
        seed=gate_config["seed"],
    )

    margin = gate_config["margin"]
    ci_passed = ci_low > margin
    verdict = "clears" if ci_passed else "does not clear"
    checks.append(
        CheckResult(
            name="bootstrap_ci",
            passed=ci_passed,
            observed=ci_low,
            threshold=margin,
            reason=f"ci_low {ci_low:.3f} {verdict} margin {margin}",
        )
    )

    return GateDecision(
        promote=ci_passed,
        checks=tuple(checks),
        delta_observed=delta_observed,
        ci_low=ci_low,
        ci_high=ci_high,
        reason=checks[-1].reason,
    )
