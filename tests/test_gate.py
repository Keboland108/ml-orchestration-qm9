"""Tests for the promotion gate.

The gate is a pure function over metrics, so tests construct synthetic
error distributions with known properties and assert on the decision.
"""

import numpy as np

from molpipe.gate import gate_decision

GATE_CONFIG = {"min_samples": 1000, "margin": 0.5, "n_boot": 1000, "seed": 7}


def _metrics(abs_errors: np.ndarray) -> dict:
    return {"mae": float(np.mean(abs_errors)), "abs_errors": abs_errors}


def get_test_input(offset: float, size=2000):
    rng = np.random.default_rng(GATE_CONFIG["seed"])
    candidate_errs = rng.uniform(5, 15, size=size)
    incumbent_errs = candidate_errs + offset

    return candidate_errs, incumbent_errs


def test_clear_winner_promotes():
    cand_errs, incu_errs = get_test_input(offset=2.0)

    decision = gate_decision(
        candidate_metrics=_metrics(cand_errs),
        incumbent_metrics=_metrics(incu_errs),
        gate_config=GATE_CONFIG,
    )
    assert decision.promote is True


# TODO(Kyle) — four more, same pattern, different offsets:
#   test_identical_models_do_not_promote        offset 0.0
#   test_real_but_below_margin_does_not_promote offset +0.1, margin 0.5
#   test_small_sample_abstains                  size=10
#   test_cold_start_promotes                    incumbent_metrics=None
#


def test_identical_models_do_not_promote():
    cand_errs, incu_errs = get_test_input(offset=0.0)

    decision = gate_decision(
        candidate_metrics=_metrics(cand_errs),
        incumbent_metrics=_metrics(incu_errs),
        gate_config=GATE_CONFIG,
    )
    assert decision.promote is False


def test_real_but_below_margin_does_not_promote():
    cand_errs, incu_errs = get_test_input(offset=0.1)

    decision = gate_decision(
        candidate_metrics=_metrics(cand_errs),
        incumbent_metrics=_metrics(incu_errs),
        gate_config=GATE_CONFIG,
    )
    assert decision.promote is False


def test_small_sample_abstains():
    cand_errs, incum_errs = get_test_input(offset=2.0, size=10)

    decision = gate_decision(
        candidate_metrics=_metrics(cand_errs),
        incumbent_metrics=_metrics(incum_errs),
        gate_config=GATE_CONFIG,
    )
    assert decision.promote is False


def test_cold_start_promotes():

    cand_errs, _ = get_test_input(offset=0.0)

    decision = gate_decision(
        candidate_metrics=_metrics(cand_errs),
        incumbent_metrics=None,
        gate_config=GATE_CONFIG,
    )
    assert decision.promote is True
