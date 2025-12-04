#!/usr/bin/env python3

import argparse
from lib.helpers import (
    get_movies_by_keyword, 
    build_idx, 
    get_tf, 
    get_idf, 
    get_tfidf,
    bm25_idf_command,
    bm25_tf_command
)
from lib.search_utils import BM25_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index from movies.json")

    tf_parser = subparsers.add_parser("tf", help="Term frequency in the document with the given ID")
    tf_parser.add_argument("id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term for which term frequency will be shown")

    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency")
    idf_parser.add_argument("term", type=str, help="term for which inverse document frequency will be shown")

    tfidf_parser = subparsers.add_parser("tfidf", help="Term Frequency-Inverse Document Frequency for a given term and document ID")
    tfidf_parser.add_argument("doc_id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="term for which Term Frequency-Inverse Document Frequency will be shown")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            get_movies_by_keyword(args.query)
        case "build":
            print("Building inverted index")
            build_idx()
        case "tf":
            tf = get_tf(args.id, args.term)
            print(f"Term frequency for {args.term} in document {args.id} is {tf}")
        case "idf":
            idf = get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()