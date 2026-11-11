"""Command line: `python -m alpine_line_exposure demo | build | rank`."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="alpine_line_exposure", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="rebuild data/ from the open sources (network, needs the [fetch] extra)")
    b.add_argument("--data", type=Path, default=ROOT / "data")
    b.add_argument("--cache", type=Path, default=ROOT / "cache")
    b.add_argument("--start", default="Andermatt")
    b.add_argument("--end", default="Disentis/Mustér")
    b.add_argument("--section-m", type=float, default=100.0)
    d = sub.add_parser("demo", help="validation + ranking + figures from the committed derived data (offline)")
    d.add_argument("--data", type=Path, default=ROOT / "data")
    d.add_argument("--out", type=Path, default=ROOT / "results")
    d.add_argument("--docs", type=Path, default=ROOT / "docs")
    d.add_argument("--top", type=int, default=10)
    d.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "build":
        from .build import run

        run(a.data, a.cache, a.start, a.end, a.section_m, log=lambda m: print(m, flush=True))
        return 0
    if a.cmd == "demo":
        from .pipeline import demo

        demo(a.data, a.out, a.docs, top_count=a.top, figures=not a.no_figures)
        return 0
    return 1
