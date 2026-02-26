#!/usr/bin/env python3
"""Task router for evaluation scripts."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from utils import eval_ambient, eval_industry, eval_person, eval_road

TASK_MODULES: Dict[str, object] = {
    "road": eval_road,
    "person": eval_person,
    "ambient": eval_ambient,
    "industry": eval_industry,
}
DEFAULT_TASK = "road"


def _build_task_parser(task: str) -> argparse.ArgumentParser:
    parser = TASK_MODULES[task].build_parser()
    parser.add_argument(
        "--task",
        choices=sorted(TASK_MODULES.keys()),
        default=task,
        help="Selects which evaluation task to run.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", choices=sorted(TASK_MODULES.keys()), default=DEFAULT_TASK)
    pre_args, _ = pre.parse_known_args(argv)

    parser = _build_task_parser(pre_args.task)
    args = parser.parse_args(argv)
    TASK_MODULES[pre_args.task].run(args)


if __name__ == "__main__":
    main()
