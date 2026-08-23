"""The promotion gate. Pure: metrics in, decision out. No I/O, no MLflow.

Non-promotion is a successful decision, never an error.
"""

from dataclasses import dataclass


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


def get_ci_range() -> tuple[float, float]:
    # TODO(Kyle): replace stub with the paired bootstrap —
    # takes both abs_errors arrays + n_boot + seed, returns the
    # 2.5th/97.5th percentiles of the resampled MAE deltas.
    return 0.6, 0.7


def gate_decision(
    candidate_metrics: dict,
    incumbent_metrics: dict | None,
    gate_config: dict,
) -> GateDecision:
    # TODO(Kyle): sample-size floor check; cold-start branch
    # (incumbent_metrics is None); build one CheckResult per check.
    ci_low, ci_high = get_ci_range()

    promote = ci_low > gate_config["margin"]
    return GateDecision(
        promote=promote,
        checks=(),
        delta_observed=None,
        ci_low=ci_low,
        ci_high=ci_high,
        reason="stub — bootstrap and guardrails not implemented",
    )
