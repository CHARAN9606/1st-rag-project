"""
10_multi_query_retrieval_free.py

Free/local Multi-Query Retrieval example.

Models:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Query generator: Qwen/Qwen2.5-1.5B-Instruct

No OpenAI API key or paid model is required.

IMPORTANT:
Your Chroma database must have been created with the same
all-MiniLM-L6-v2 embedding model. If it was created with
OpenAIEmbeddings, rebuild the database first.
"""

import gc
from typing import List

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================================================================
# 1. CONFIGURATION
# ================================================================

PERSISTENT_DIRECTORY = "db/chroma_db"

# Free CPU embedding model for semantic search.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Free Hugging Face instruction model for creating alternative queries.
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

NUMBER_OF_QUERY_VARIATIONS = 3
DOCUMENTS_PER_QUERY = 5

# Change this question to test your own documents.
ORIGINAL_QUERY = "How does Tesla make money?"


# ================================================================
# 2. LOAD FREE EMBEDDINGS
# ================================================================

print("=" * 70)
print("LOADING FREE HUGGING FACE EMBEDDINGS")
print("=" * 70)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print(f"Model: {EMBEDDING_MODEL_NAME}")
print("Device: CPU")
print("Embeddings loaded successfully.")


# ================================================================
# 3. LOAD CHROMA DATABASE
# ================================================================

print("\n" + "=" * 70)
print("LOADING CHROMA DATABASE")
print("=" * 70)

db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

print("Chroma database loaded successfully.")


# ================================================================
# 4. LOAD FREE QWEN MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING FREE QWEN MODEL")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"Qwen loaded on GPU: {torch.cuda.get_device_name(0)}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.float32,
    )
    print("Qwen loaded on CPU.")

model.eval()


# ================================================================
# 5. QWEN TEXT GENERATION
# ================================================================

def generate_text(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text from Qwen using the supplied prompt."""

    messages = [
        {
            "role": "system",
            "content": (
                "You generate concise alternative search queries "
                "for a retrieval system."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[-1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    del inputs
    del outputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return answer


# ================================================================
# 6. GENERATE QUERY VARIATIONS
# ================================================================

def generate_query_variations(
    original_query: str,
    number_of_queries: int = 3,
) -> List[str]:
    """
    Generate several alternative versions of the original question.
    Different wording can retrieve different relevant chunks.
    """

    prompt = f"""
Generate exactly {number_of_queries} different search queries
for the question below.

Original question:
{original_query}

Rules:
- Keep the same meaning.
- Approach the question from different angles.
- Make every query useful for document retrieval.
- Return ONLY the queries.
- Put one query on each line.
- Do not number the lines.
"""

    response = generate_text(
        prompt,
        max_new_tokens=160,
    )

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    cleaned_queries = []

    for line in lines:
        # Remove common accidental numbering: "1. query", "2) query".
        if len(line) >= 3 and line[0].isdigit() and line[1] in ".)":
            line = line[2:].strip()
        elif len(line) >= 4 and line[:2].isdigit() and line[2] in ".)":
            line = line[3:].strip()

        if line:
            cleaned_queries.append(line)

    # Include the original query so retrieval is never dependent
    # entirely on the generated variations.
    final_queries = [original_query]

    for query in cleaned_queries:
        if query.lower() != original_query.lower():
            final_queries.append(query)

        if len(final_queries) >= number_of_queries + 1:
            break

    return final_queries


# ================================================================
# 7. RETRIEVE DOCUMENTS FOR EACH QUERY
# ================================================================

def retrieve_for_queries(
    queries: List[str],
    k: int = 5,
) -> List[List[Document]]:
    """Run semantic retrieval separately for every query."""

    retriever = db.as_retriever(
        search_kwargs={"k": k},
    )

    all_results = []

    for index, query in enumerate(queries, start=1):

        print("\n" + "-" * 70)
        print(f"QUERY {index}")
        print("-" * 70)
        print(query)

        documents = retriever.invoke(query)
        all_results.append(documents)

        print(f"Retrieved {len(documents)} documents.")

        for doc_index, document in enumerate(
            documents,
            start=1,
        ):
            preview = (
                document.page_content
                .replace("\n", " ")
                [:200]
            )

            print(
                f"  Document {doc_index}: "
                f"{preview}..."
            )

    return all_results


# ================================================================
# 8. REMOVE DUPLICATE DOCUMENTS
# ================================================================

def deduplicate_documents(
    all_results: List[List[Document]],
) -> List[Document]:
    """Combine results and remove duplicate chunks."""

    unique_documents = []
    seen_content = set()

    for documents in all_results:
        for document in documents:

            content = document.page_content.strip()

            if not content:
                continue

            if content in seen_content:
                continue

            seen_content.add(content)
            unique_documents.append(document)

    return unique_documents


# ================================================================
# 9. MAIN PROGRAM
# ================================================================

def main():

    print("\n" + "=" * 70)
    print("MULTI-QUERY RETRIEVAL")
    print("=" * 70)

    print(f"\nOriginal query: {ORIGINAL_QUERY}")

    # Step 1: Generate alternative search queries.
    print("\n" + "=" * 70)
    print("STEP 1: GENERATING QUERY VARIATIONS")
    print("=" * 70)

    query_variations = generate_query_variations(
        ORIGINAL_QUERY,
        NUMBER_OF_QUERY_VARIATIONS,
    )

    for index, query in enumerate(query_variations, start=1):
        print(f"{index}. {query}")

    # Step 2: Search Chroma using every query.
    print("\n" + "=" * 70)
    print("STEP 2: RETRIEVING DOCUMENTS")
    print("=" * 70)

    all_results = retrieve_for_queries(
        query_variations,
        DOCUMENTS_PER_QUERY,
    )

    # Step 3: Combine the results and remove duplicates.
    print("\n" + "=" * 70)
    print("STEP 3: COMBINING RESULTS")
    print("=" * 70)

    unique_documents = deduplicate_documents(all_results)

    print(
        f"Unique documents after deduplication: "
        f"{len(unique_documents)}"
    )

    # Step 4: Display the final retrieval set.
    print("\n" + "=" * 70)
    print("FINAL MULTI-QUERY RESULTS")
    print("=" * 70)

    for index, document in enumerate(
        unique_documents,
        start=1,
    ):
        print(f"\nDocument {index}")
        print("-" * 60)
        print(document.page_content[:500])

        if document.metadata:
            print("\nMetadata:")
            print(document.metadata)

    print("\n" + "=" * 70)
    print("MULTI-QUERY RETRIEVAL COMPLETE")
    print("=" * 70)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
