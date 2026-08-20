"""Ingest one new quarterly CSBC file and refresh the analysis.

Two modes, because a new file answers two different questions:

  standalone  what does this quarter say on its own?   (no history needed)
  append      how does the whole series look now?      (appends to the history)
  both        run standalone first, then append        (default)

Examples
--------
    python src/refresh_quarter.py --new-file "data/raw/Data CSBC-Q3 2024.csv"
    python src/refresh_quarter.py --new-file new.csv --mode standalone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src/ is not an installed package, so make this script runnable from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csbc  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a new quarterly CSBC file and refresh the analysis.",
    )
    parser.add_argument("--new-file", required=True,
                       help="Path to the new quarterly CSV, in the supplied format.")
    parser.add_argument("--mode", choices=["standalone", "append", "both"], default="both",
                       help="standalone: analyse the quarter alone. append: add it to the "
                            "combined history and refresh the trend. both (default): each in turn.")
    parser.add_argument("--combined-file", default=str(csbc.PROCESSED_FILE),
                       help="Combined history CSV; created if it does not exist.")
    parser.add_argument("--output-dir", default=str(csbc.OUTPUT_DIR),
                       help="Where figures and metric tables are written.")
    args = parser.parse_args(argv)

    new_file = Path(args.new_file)
    if not new_file.exists():
        parser.error(f"No such file: {new_file}")

    csbc.ensure_dirs()
    csbc.apply_chart_theme()
    output_dir = Path(args.output_dir)

    # Validation failures are expected operating conditions, not crashes: report the
    # reason on stderr and exit non-zero so a scheduler can act on it.
    try:
        if args.mode in ("standalone", "both"):
            print("=" * 72)
            print("MODE 1 - standalone: this quarter on its own")
            print("=" * 72)
            csbc.analyse_single_quarter(new_file, output_dir=output_dir)

        if args.mode in ("append", "both"):
            print()
            print("=" * 72)
            print("MODE 2 - append: refresh the combined history")
            print("=" * 72)
            csbc.ingest_and_refresh(new_file, combined_file=Path(args.combined_file),
                                   output_dir=output_dir)
    except ValueError as error:
        print(f"\nREFUSED: {error}", file=sys.stderr)
        print("Nothing was written. Review the file, then re-run.", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
