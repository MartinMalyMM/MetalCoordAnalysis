"""Regression tests for ``--max_size`` / ``Config().max_sample_size``.

Background
----------
``--max_size`` limits how many COD reference structures are sampled per class
in :mod:`metalCoord.analysis.stats`. The option worked up to and including
v0.2.11, but the CLI refactor in v0.2.13 ("refactor CLI into handler modules")
hoisted the command-handler imports to module top level. As a result
``metalCoord.analysis.stats`` is now imported *before* the CLI sets
``Config().max_sample_size``. Because the limit is captured in a module-level
constant::

    MAX_FILES = Config().max_sample_size if Config().max_sample_size else 2000

at import time, the user value is ignored and every run samples up to 2000
references regardless of ``--max_size``.

Design notes
------------
These tests deliberately observe the *limit itself* rather than the ``count``
field of the JSON output, because:

* observing the output is fragile and breaks if the output is disabled or its
  format changes; and
* asserting ``count <= max_size`` is *vacuous* whenever a class simply has
  fewer references than ``max_size`` -- it passes even with a broken cap.

They are marked ``xfail(strict=True)``: on the current (buggy) build they
document the regression and are expected to fail. Once the production fix lands
they will XPASS, and strict mode turns an unexpected pass into a failure so the
markers get removed together with the fix.
"""

import os

import numpy as np
import pytest

from metalCoord.config import Config

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_3KW8 = os.path.join(tests_dir, "data", "models", "3kw8.cif")


def _effective_max_files(stats_module):
    """Return the reference-sampling cap the way the analysis actually applies it.

    Works against both the current code (module-level constant ``MAX_FILES``)
    and the intended fix (a ``_max_files()`` helper that reads ``Config`` at
    call time), so the same assertions stay valid before and after the fix.
    """
    helper = getattr(stats_module, "_max_files", None)
    if callable(helper):
        return helper()
    return stats_module.MAX_FILES


@pytest.fixture
def restore_sample_size():
    """Snapshot and restore the singleton ``Config`` sample-size fields."""
    cfg = Config()
    saved = (cfg.min_sample_size, cfg.max_sample_size)
    yield cfg
    cfg.min_sample_size, cfg.max_sample_size = saved


def test_default_max_files_is_2000(restore_sample_size):
    """Without ``--max_size`` the cap defaults to 2000 (unchanged behaviour)."""
    import metalCoord.analysis.stats as stats_module

    restore_sample_size.max_sample_size = None
    assert _effective_max_files(stats_module) == 2000


@pytest.mark.xfail(
    reason="Regression v0.2.13+: max_size is frozen at import time; "
    "remove this marker once stats.py reads the cap at call time.",
    strict=True,
)
def test_max_size_honoured_after_import(restore_sample_size):
    """A ``--max_size`` set after import must change the effective cap.

    This mirrors the production import order: ``stats.py`` is imported at pytest
    collection time, i.e. *before* ``Config().max_sample_size`` is assigned here
    -- exactly the situation that the v0.2.13 import refactor created.
    """
    import metalCoord.analysis.stats as stats_module

    restore_sample_size.max_sample_size = 100
    assert _effective_max_files(stats_module) == 100


@pytest.mark.xfail(
    reason="Regression v0.2.13+: max_size never reaches the sampling step; "
    "remove this marker once stats.py reads the cap at call time.",
    strict=True,
)
def test_max_size_truncates_reference_pool(restore_sample_size, monkeypatch, tmp_path):
    """``--max_size`` must actually truncate the candidate reference pool.

    Spies on :func:`numpy.random.choice`, which ``stats.py`` uses to subsample
    references when a class has more candidates than the cap. The 3kw8/CU site
    has a class with >300 candidate references, so a cap of 50 must trigger a
    ``choice(files, 50, ...)`` call. On the buggy build the cap stays at 2000,
    the pool (~328) never exceeds it, and ``choice`` is never invoked for the
    cap -- so no matching call is recorded.

    This runs the analysis in-process and inspects the call, so it does not
    depend on the JSON output being produced or on its format.
    """
    from metalCoord.service.analysis import get_stats

    cfg = restore_sample_size
    cfg.min_sample_size = 5
    cfg.max_sample_size = 50

    recorded = []
    real_choice = np.random.choice

    def spy(a, size=None, *args, **kwargs):
        try:
            pool_size = len(a)
        except TypeError:
            pool_size = None
        recorded.append((pool_size, size))
        return real_choice(a, size, *args, **kwargs)

    monkeypatch.setattr(np.random, "choice", spy)

    output = str(tmp_path / "3kw8_CU.json")
    get_stats("CU", MODEL_3KW8, output)

    # A cap-driven truncation: a pool larger than the cap, sampled down to the cap.
    assert any(
        size == 50 and (pool_size is None or pool_size > 50)
        for pool_size, size in recorded
    ), f"max_size=50 never truncated the reference pool; choice calls: {recorded}"
