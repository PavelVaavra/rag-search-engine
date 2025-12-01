#!/usr/bin/env python3

import argparse
from lib.helpers import get_movies_by_keyword, build_idx, get_tf

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index from movies.json")

    tf_parser = subparsers.add_parser("tf", help="Term frequency in the document with the given ID")
    tf_parser.add_argument("id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term for which term frequency will be shown")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            get_movies_by_keyword(args.query)
        case "build":
            print("Building inverted index")
            build_idx()
        case "tf":
            get_tf(args.id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()