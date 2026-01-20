import argparse
import json

from lib.search_utils import GOLDEN_DATASET_PATH, DEFAULT_RRF_K
from lib.hybrid_search import rrf_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(GOLDEN_DATASET_PATH, "r") as file:
        golden_dataset = json.load(file)
    # For each test case in the golden dataset, run RRF search. RRF's k parameter should be 60, and the "top k" (number of results) should be the value of the --limit flag... Sorry about all the Ks; it's a bit confusing.
    # Based on the formula above (the percentage of titles in the results that are found in the golden dataset), calculate the precision for each test case.
    # Print the results in this format:
    # precision = relevant_retrieved / total_retrieved

    print(f"k={limit}\n")
    
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant = test_case["relevant_docs"]
        # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
        retrieved = []
        relevant_retrieved = 0
        docs = rrf_search(query, DEFAULT_RRF_K, limit)
        for _, lst in docs.items():
            title = lst[3]
            retrieved.append(title)
            if title in relevant:
                relevant_retrieved += 1
        total_retrieved = len(retrieved)
        precision = relevant_retrieved / total_retrieved
        total_relevant = len(relevant)
        recall = relevant_retrieved / total_relevant
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {", ".join(retrieved)}")
        print(f"  - Relevant: {", ".join(relevant)}")

if __name__ == "__main__":
    main()