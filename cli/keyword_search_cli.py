#!/usr/bin/env python3

import argparse
import json
import string

def get_movies(keyword):
    translator = str.maketrans("", "", string.punctuation)

    keyword = keyword.lower().translate(translator)
    keyword_tokens = [word for word in keyword if word != ""]
    
    with open("data/movies.json", "r") as file:
        movies = json.load(file)["movies"]
        found_movies = []

        for movie in movies:
            movie_modified = movie["title"].lower().translate(translator)
            movie_tokens = [m for m in movie_modified if m != ""]
            
            # for movie_token in movie_tokens:
            if keyword in movie_modified:
                found_movies.append(movie["title"])
                    # break
            if len(found_movies) == 5:
                break
        
        for i, movie in enumerate(found_movies):
            print(f"{i + 1} {movie}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            get_movies(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()