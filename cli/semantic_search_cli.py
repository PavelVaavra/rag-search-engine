#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search
)

from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model")

    embedding_parser = subparsers.add_parser("embed_text", help="Create an embedding from the text")
    embedding_parser.add_argument("text", type=str, help="text which an embedding will be created from")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embedding_query_parser = subparsers.add_parser("embedquery", help="Create an embedding from the query")
    embedding_query_parser.add_argument("query", type=str, help="query which an embedding will be created from")

    search_parser = subparsers.add_parser("search", help="Search movies semantically")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="How many top documents should be shown")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()