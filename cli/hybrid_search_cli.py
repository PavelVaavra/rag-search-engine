import argparse

from lib.hybrid_search import normalize


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="min-max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()