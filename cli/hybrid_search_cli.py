import argparse

from lib.hybrid_search import normalize, weighted_search

from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="min-max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search combination of the keyword and semantic search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weighting constant")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="How many top documents should be shown")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize(args.scores)
            for normalize_score in normalized_scores:
                print(f"* {normalize_score:.4f}")

        case "weighted-search":
            docs = weighted_search(args.query, args.alpha, args.limit)
            for i, item in enumerate(docs.items()):
                key = item[0]
                title = item[1][3]
                hybrid_score = item[1][2]
                keyword_score = item[1][0]
                semantic_score = item[1][1]
                description = item[1][4]
                print(f"{i + 1}. {title}")
                print(f"Hybrid Score: {hybrid_score:.3f}")
                print(f"BM25: {keyword_score:.3f}, Semantic: {semantic_score:.3f}")
                print(f"{description}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()