"""Frozen legacy results supplement, never replace, independent physics tests."""
import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from tests.reference_cases import compute_reference_cases


def test_frozen_legacy_reference_cases():
    reference = json.loads((Path(__file__).parent/"reference"/"legacy_cpu_x64.json").read_text())
    actual = compute_reference_cases()
    assert set(actual) == set(reference["cases"])
    for name, value in actual.items():
        entry = reference["cases"][name]
        expected = np.asarray(entry["real"])
        if "imag" in entry:
            expected = expected+1j*np.asarray(entry["imag"])
        assert value.shape == expected.shape, name
        assert np.isfinite(value).all(), name
        assert_allclose(value, expected, rtol=entry["rtol"], atol=entry["atol"], err_msg=name)
