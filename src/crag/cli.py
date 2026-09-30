"""Command-line entry point.

python -m crag "Batch normalization vs layer normalization"
python -m crag "..." --trace          # show verdict, scores, kept strips
python -m crag "..." --rebuild-index  # re-embed the PDFs in data/
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from .config import Settings
from .graph import build_default_components, build_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crag", description="Corrective RAG over your PDFs.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--trace", action="store_true", help="Print intermediate pipeline state.")
    parser.add_argument("--rebuild-index", action="store_true", help="Re-embed the PDFs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    load_dotenv()

    settings = Settings.from_env()
    app = build_graph(build_default_components(settings, args.rebuild_index), settings)
    res = app.invoke({"question": args.question})

    if args.trace:
        print(f"VERDICT : {res['verdict']}")
        print(f"REASON  : {res['reason']}")
        print(f"SCORES  : {[round(s, 2) for s in res.get('scores', [])]}")
        if res.get("web_query"):
            print(f"WEBQUERY: {res['web_query']}")
        print(f"STRIPS  : kept {len(res.get('kept_strips', []))} of {len(res.get('strips', []))}")
        print("-" * 60)

    print(res["answer"])
    if res.get("sources"):
        print("\nSources:")
        for s in res["sources"]:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
