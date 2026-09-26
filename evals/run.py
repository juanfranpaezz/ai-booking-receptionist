"""Zero-install entry point for the eval runner: ``python evals/run.py``.

The real implementation lives in the package
(``src/booking_receptionist/evals_runner.py``) so it is typed, importable and
exposed as the ``booking-evals`` console script. This file only puts ``src/`` on
``sys.path`` so a reviewer can run the evals straight after ``git clone``,
without ``pip install -e .``.

Exit code: 0 when all cases pass, 1 when any case fails, 2 on a bad cases file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from booking_receptionist.evals_runner import main  # noqa: E402  (path set above)

if __name__ == "__main__":
    raise SystemExit(main())
