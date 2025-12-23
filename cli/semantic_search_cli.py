#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
    chunk,
    semantic_chunk,
    semantic_chunk_print,
    embed_chunks
)

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE
)

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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="How many words should overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE, help="Max chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="How many sentences should overlap")

    subparsers.add_parser("embed_chunks", help="Embed chunks")

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
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            semantic_chunk_print(chunks, len(args.text))
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()