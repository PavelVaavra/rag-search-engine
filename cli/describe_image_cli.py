import argparse
import mimetypes
import os

from lib.search_utils import PROJECT_ROOT
from gemini_api import rewrite_query

def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")

    parser.add_argument("--image", type=str, help="the path to an image file")
    parser.add_argument("--query", type=str, help="a text query to rewrite based on the image")
    
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(os.path.join(PROJECT_ROOT, args.image), "rb") as file:
        img = file.read()

    response = rewrite_query(img, args.query, mime)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()