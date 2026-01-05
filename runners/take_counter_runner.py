from __future__ import annotations

from typing import Literal


Mode = Literal["batch", "streaming", "eval_streaming"]


def run(mode: Mode = "batch") -> int:
    """
    Unified entry point for take-counter experiments.

    - mode="batch": pre-loaded video inference runner
    - mode="streaming": streaming inference runner
    """
    if mode == "batch":
        from runners.take_counter_batch import main as _main

        return int(_main())
    if mode == "streaming":
        from runners.take_counter_streaming import run_streaming_demo as _main

        return int(_main())
    if mode == "eval_streaming":
        from runners.take_counter_streaming_eval import run_default_eval as _main

        return int(_main())
    raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    raise SystemExit(run("batch"))
