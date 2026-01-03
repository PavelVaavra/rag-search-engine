import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# client = genai.Client(api_key=api_key)

# response = client.models.generate_content(
#     # model="gemini-2.0-flash-001",
#     model="gemini-2.5-flash-lite", 
#     contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
# )
# print(response.text)
# print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
# print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")

def spell_prompt(query):
    return f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

def enhance(method, query):
    client = genai.Client(api_key=api_key)
    
    match method:
        case "spell":
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=spell_prompt(query)
            )
            enhanced_query = response.text

    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")

    return enhanced_query