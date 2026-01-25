import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Perform verification of an image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Path to an image to be embedded")

    image_search_parser = subparsers.add_parser("image_search", help="Search image through documents")
    image_search_parser.add_argument("image_path", type=str, help="Path to an image to be searched for")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            found_docs = image_search_command(args.image_path)

            for i in range(len(found_docs)):
                title = found_docs[i]["title"]
                similarity = found_docs[i]["similarity_score"]
                description = found_docs[i]["description"][:100]
                print(f"{i + 1}. {title} (similarity: {similarity:.3f})")
                print(f"   {description}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()