"""
9_retrieval_methods_free.py

Free/local version of the retrieval-methods example.

This file removes the paid OpenAI embedding model and uses the free
Hugging Face sentence-transformers model:

    sentence-transformers/all-MiniLM-L6-v2

IMPORTANT:
The Chroma database must have been created with the SAME embedding model.
If your existing db/chroma_db was created with OpenAIEmbeddings, do not
reuse it with MiniLM. Rebuild the database with the same MiniLM embedding
model first.

Install once in your virtual environment:

    pip install -U langchain-chroma langchain-huggingface sentence-transformers

The three retrieval methods demonstrated here are:
1. Basic similarity search
2. Similarity search with a score threshold
3. Maximum Marginal Relevance (MMR)
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ================================================================
# 1. CONFIGURATION
# ================================================================

# This must point to the Chroma database created with the
# same Hugging Face embedding model used below.
PERSISTENT_DIRECTORY = "db/chroma_db"

# Free local embedding model.
# It runs on CPU by default, so it does not require a GPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of documents returned by the retrievers.
TOP_K = 3

# Query used to test the retrieval methods.
QUERY = "When did microsoft had started?"

# You can try another question, for example:
# QUERY = "How do you plant tomatoes in a garden?"


# ================================================================
# 2. LOAD THE FREE HUGGING FACE EMBEDDING MODEL
# ================================================================

print("Loading free Hugging Face embedding model...")
print(f"Model: {EMBEDDING_MODEL_NAME}")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding model loaded successfully.")


# ================================================================
# 3. LOAD CHROMA DATABASE
# ================================================================

print("\nLoading Chroma database...")

db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={
        "hnsw:space": "cosine"
    }
)

print("Chroma database loaded successfully.")
print(f"\nQuery: {QUERY}\n")


# ================================================================
# HELPER FUNCTION
# ================================================================

def print_documents(documents, title):
    """
    Print the documents returned by a retrieval method.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Retrieved {len(documents)} documents:\n")

    if not documents:
        print("No documents were returned.")
        return

    for index, document in enumerate(documents, start=1):

        print(f"Document {index}")
        print("-" * 50)

        print(document.page_content)

        if document.metadata:
            print("\nMetadata:")
            print(document.metadata)

        print()


# ================================================================
# METHOD 1: BASIC SIMILARITY SEARCH
# ================================================================

"""
Basic similarity search returns the documents that are most
semantically similar to the query.

Use this when:
- You want the top few relevant chunks.
- You do not need a minimum relevance threshold.
"""

# print("\n" + "=" * 70)
# print("METHOD 1: BASIC SIMILARITY SEARCH")
# print("=" * 70)

# similarity_retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": TOP_K
#     }
# )

# similarity_docs = similarity_retriever.invoke(QUERY)

# print_documents(
#     similarity_docs,
#     "Similarity Search Results"
# )


# ================================================================
# METHOD 2: SIMILARITY SEARCH WITH SCORE THRESHOLD
# ================================================================

"""
This method returns documents only when their relevance score
passes the configured threshold.

A threshold of 0.3 is a starting point, not a universal value.
You may need to tune it for your own documents and embedding model.

Use this when:
- You would rather return fewer documents than weak matches.
"""

# print("\n" + "=" * 70)
# print("METHOD 2: SIMILARITY SEARCH WITH SCORE THRESHOLD")
# print("=" * 70)

# THRESHOLD = 0.30

# threshold_retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": TOP_K,
#         "score_threshold": THRESHOLD
#     }
# )

# threshold_docs = threshold_retriever.invoke(QUERY)

# print(
#     f"Score threshold: {THRESHOLD}"
# )

# print_documents(
#     threshold_docs,
#     "Similarity + Score Threshold Results"
# )


# ================================================================
# METHOD 3: MAXIMUM MARGINAL RELEVANCE (MMR)
# ================================================================

"""
MMR tries to balance:

    relevance to the query
                +
    diversity between returned documents

This helps reduce cases where all retrieved chunks contain
almost the same information.

Parameters:
- k = number of final documents
- fetch_k = number of candidates considered initially
- lambda_mult:
      1.0 -> more relevance
      0.0 -> more diversity
      0.5 -> balanced starting point
"""

print("\n" + "=" * 70)
print("METHOD 3: MAXIMUM MARGINAL RELEVANCE (MMR)")
print("=" * 70)

mmr_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": TOP_K,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

mmr_docs = mmr_retriever.invoke(QUERY)

print_documents(
    mmr_docs,
    "MMR Results"
)


# ================================================================
# OPTIONAL: COMPARE ALL THREE METHODS
# ================================================================

print("\n" + "=" * 70)
print("RETRIEVAL COMPARISON")
print("=" * 70)

print(f"Query: {QUERY}\n")

# print(
#     f"1. Similarity search       : "
#     f"{len(similarity_docs)} documents"
# )

# print(
#     f"2. Score threshold ({THRESHOLD}): "
#     f"{len(threshold_docs)} documents"
# )

print(
    f"3. MMR                     : "
    f"{len(mmr_docs)} documents"
)

print("\nDone!")
print(
    "Try changing QUERY, TOP_K, THRESHOLD, and lambda_mult "
    "to compare retrieval behavior."
)
