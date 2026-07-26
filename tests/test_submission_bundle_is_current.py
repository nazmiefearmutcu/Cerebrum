"""Guard for the conftest.py import rebinding.

`conftest.py` replaces the `cerebrum` package with the generated single-file bundle
`cerebrum_submission.py` for the whole test run. That means every other test in this
suite validates the BUNDLE, not the `cerebrum/` package directory. The two are only
interchangeable while the bundle is byte-for-byte what `build_submission.py` produces
from the current package.

If this test fails, the bundle has drifted: the package has been edited without
regenerating the bundle, and the rest of the suite is silently testing stale code.
Fix it by running `python3 build_submission.py`.
"""

import difflib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from build_submission import BUNDLE_PATH, build_bundle  # noqa: E402


def test_bundle_matches_package():
    bundle_file = os.path.join(REPO_ROOT, BUNDLE_PATH)
    assert os.path.exists(bundle_file), (
        f"{BUNDLE_PATH} is missing, but conftest.py rebinds the `cerebrum` package to it. "
        "Run `python3 build_submission.py`."
    )

    with open(bundle_file, "r") as f:
        committed = f.read()

    expected = build_bundle(REPO_ROOT)

    if committed != expected:
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    committed.splitlines(),
                    expected.splitlines(),
                    fromfile=f"{BUNDLE_PATH} (committed)",
                    tofile=f"{BUNDLE_PATH} (rebuilt from cerebrum/)",
                    lineterm="",
                )
            )[:200]
        )
        raise AssertionError(
            "BUNDLE DRIFT: cerebrum_submission.py no longer matches the cerebrum/ package.\n"
            "conftest.py points the whole test suite at this bundle, so every other test is\n"
            "currently exercising stale code. Regenerate it with `python3 build_submission.py`.\n\n"
            f"First differences:\n{diff}"
        )
