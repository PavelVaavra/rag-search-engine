import argparse

from lib.hybrid_search import rrf_search
from lib.search_utils import DEFAULT_RRF_K, DEFAULT_SEARCH_LIMIT
from gemini_api import rag


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            
            # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
            docs = rrf_search(query, DEFAULT_RRF_K, DEFAULT_SEARCH_LIMIT)

            rag_response = rag(docs, query)

            print("Search Results:")
            for _, lst in docs.items():
                print(f"  - {lst[3]}")
            print(f"RAG Response:\n{rag_response}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()