import os
import json
from time import sleep
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

def rewrite_prompt(query):
    return f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

def expand_prompt(query):
    return f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

def enhance(method, query):
    client = genai.Client(api_key=api_key)
    
    match method:
        case "spell":
            prompt = spell_prompt(query)
        case "rewrite":
            prompt = rewrite_prompt(query)
        case "expand":
            prompt = expand_prompt(query)
            
    response = client.models.generate_content(
         model="gemini-2.5-flash-lite",
         contents=prompt
    )
    enhanced_query = response.text

    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")

    return enhanced_query

def rerank_individual(docs, query):
    # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
    client = genai.Client(api_key=api_key)

    for _, lst in docs.items():
        title = lst[3]
        description = lst[4]
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {description}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        lst.append(response.text)
        sleep(10)
    
    docs = dict(sorted(docs.items(), key=lambda item: int(item[1][5]), reverse=True))

    return docs

def rerank_batch(docs, query):
    # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
    client = genai.Client(api_key=api_key)

    doc_list = []
    for id, lst in docs.items():
        title = lst[3]
        description = lst[4][:200]
        movie = f"{id}: {title} - {description}"
        doc_list.append(movie)

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    print(response.text)
    order = json.loads(response.text.strip())
    
    docs_sorted = {}
    for i in range(len(order)):
        id = order[i]
        docs_sorted[id] = docs[id]
    
    return docs_sorted

def evaluate(docs, query):
    # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
    client = genai.Client(api_key=api_key)

    doc_list = []
    for id, lst in docs.items():
        title = lst[3]
        description = lst[4]    #[:200]
        movie = f"{title}: {description}"
        doc_list.append(movie)

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{doc_list_str}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    # print(response.text)
    return json.loads(response.text.strip())

def rag(docs, query):
    # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
    client = genai.Client(api_key=api_key)

    doc_list = []
    for _, lst in docs.items():
        title = lst[3]
        description = lst[4]    #[:200]
        movie = f"{title}: {description}"
        doc_list.append(movie)

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{doc_list_str}

Provide a comprehensive answer that addresses the query:"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text

def summarize(docs, query):
    # { id: [keyword_score, semantic_score, hybrid_score, title, description] }
    client = genai.Client(api_key=api_key)

    doc_list = []
    for _, lst in docs.items():
        title = lst[3]
        description = lst[4]
        movie = f"{title}: {description}"
        doc_list.append(movie)

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{doc_list_str}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text
