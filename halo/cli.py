"""Console-script entrypoints declared in pyproject.toml."""
from __future__ import annotations

import sys


def trace_main() -> None:
    """``halo-trace`` → forwards to ``scripts.extract_attention_trace``."""
    from scripts.extract_attention_trace import main  # type: ignore[import-not-found]
    sys.exit(main())


def eval_main() -> None:
    """``halo-eval`` → forwards to ``scripts.run_longbench``."""
    from scripts.run_longbench import main  # type: ignore[import-not-found]
    sys.exit(main())
