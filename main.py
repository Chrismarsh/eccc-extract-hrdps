from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, Tuple

from convert import extract_day


def extract_from_pickle(pickle_path: Path) -> None:
    """Run `extract_day` for every (year, month, day) tuple stored in a pickle file."""
    with pickle_path.open("rb") as f:
        loaded_array = pickle.load(f)

    for year, month, day in loaded_array:
        extract_day(year, month, day)


def extract_from_cli_triplets(parts: Iterable[str]) -> None:
    """Parse command-line triplets (year, month, day) and run `extract_day` for each one."""
    parts = list(parts)
    if len(parts) % 3 != 0 or not parts:
        raise ValueError("Provide dates as YEAR MONTH DAY triplets (e.g., 2024 07 15).")

    def as_ints(triplet: Tuple[str, str, str]) -> Tuple[int, int, int]:
        try:
            return tuple(int(value) for value in triplet)  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"Triplet {triplet} must contain integer values.") from exc

    for idx in range(0, len(parts), 3):
        year, month, day = as_ints((parts[idx], parts[idx + 1], parts[idx + 2]))
        extract_day(year, month, day)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HRDPS extraction for one or more dates."
    )
    parser.add_argument(
        "--file",
        dest="pickle_path",
        type=Path,
        help="Optional pickle file containing (year, month, day) tuples.",
    )
    parser.add_argument(
        "date_parts",
        nargs="*",
        help="Dates provided as YEAR MONTH DAY triplets (e.g., 2024 07 15).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)

    if args.pickle_path is not None:
        extract_from_pickle(args.pickle_path)
        return

    if not args.date_parts:
        raise ValueError("No dates provided. Use --file or supply YEAR MONTH DAY triplets.")

    extract_from_cli_triplets(args.date_parts)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
