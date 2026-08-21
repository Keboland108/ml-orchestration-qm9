"""Smoke test — keeps CI green until real tests (gate/validation) land.

Replace/extend with the first real test; the gate is the primary test target.
"""

import marlabs


def test_package_imports_and_has_version():
    assert marlabs.__version__
