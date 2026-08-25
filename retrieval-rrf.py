"""Free Multi-Query Retrieval with Reciprocal Rank Fusion (RRF)."""
import gc
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSISTENT_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUMBER_OF_QUERY_VARIATIONS = 3
DOCUMENTS_PER_QUERY = 5
FINAL_DOCUMENT_COUNT = 5
RRF_K = 60
ORIGINAL_QUERY = "in which year was nvidia founded?"

print("Loading free Hugging Face embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading Chroma database...")
db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

print("Loading free Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    print(f"Qwen loaded on GPU: {torch.cuda.get_device_name(0)}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, dtype=torch.float32
    )
    print("Qwen loaded on CPU.")
model.eval()


def generate_text(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text with Qwen."""
    messages = [
        {"role": "system", "content": "Generate concise alternative search queries for document retrieval."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0, inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def generate_query_variations(original_query: str, number_of_queries: int = 3) -> List[str]:
    """Create alternative versions of the user's question."""
    prompt = f"""
Generate exactly {number_of_queries} different search queries for this question.
Question: {original_query}
Rules: keep the meaning, use different wording, return only one query per line, no numbering.
"""
    response = generate_text(prompt, 160)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        if len(line) >= 3 and line[0].isdigit() and line[1] in ".)":
            line = line[2:].strip()
        elif len(line) >= 4 and line[:2].isdigit() and line[2] in ".)":
            line = line[3:].strip()
        if line:
            cleaned.append(line)
    queries = [original_query]
    for query in cleaned:
        if query.lower() != original_query.lower():
            queries.append(query)
        if len(queries) >= number_of_queries + 1:
            break
    return queries


def retrieve_ranked_chunks(queries: List[str], k: int) -> List[List[Document]]:
    """Retrieve ranked chunks for each query; rank 1 is the best result."""
    retriever = db.as_retriever(search_kwargs={"k": k})
    all_results = []
    for number, query in enumerate(queries, 1):
        print(f"\nQuery {number}: {query}")
        documents = retriever.invoke(query)
        all_results.append(documents)
        for rank, doc in enumerate(documents, 1):
            preview = doc.page_content.replace("\n", " ")[:180]
            print(f"  Rank {rank}: {preview}...")
    return all_results


def get_chunk_id(document: Document) -> str:
    """Create a stable key so the same chunk can be fused across queries."""
    metadata = document.metadata or {}
    for key in ("id", "doc_id", "chunk_id"):
        if metadata.get(key) is not None:
            return f"{key}:{metadata[key]}|{document.page_content}"
    return document.page_content.strip()


def reciprocal_rank_fusion(
    ranked_results: List[List[Document]], rrf_k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Combine ranked chunk lists using RRF.

    score(chunk) = sum(1 / (rrf_k + rank))

    A chunk that appears near the top for several query variations
    receives a higher combined score.
    """
    scores: Dict[str, float] = defaultdict(float)
    documents_by_id: Dict[str, Document] = {}

    for documents in ranked_results:
        for rank, document in enumerate(documents, start=1):
            chunk_id = get_chunk_id(document)
            documents_by_id[chunk_id] = document
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(documents_by_id[chunk_id], score) for chunk_id, score in ranked]


def main():
    print("\n" + "=" * 70)
    print("MULTI-QUERY RETRIEVAL + RECIPROCAL RANK FUSION")
    print("=" * 70)
    print(f"Original question: {ORIGINAL_QUERY}")

    print("\n1. Generating query variations...")
    queries = generate_query_variations(ORIGINAL_QUERY, NUMBER_OF_QUERY_VARIATIONS)
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")

    print("\n2. Retrieving chunks for every query...")
    ranked_results = retrieve_ranked_chunks(queries, DOCUMENTS_PER_QUERY)

    print("\n3. Applying RRF...")
    print(f"RRF constant k = {RRF_K}")
    fused_results = reciprocal_rank_fusion(ranked_results, RRF_K)
    final_results = fused_results[:FINAL_DOCUMENT_COUNT]

    print("\n" + "=" * 70)
    print("FINAL RRF-RANKED CHUNKS")
    print("=" * 70)
    print(f"Unique chunks: {len(fused_results)}")
    print(f"Final chunks selected: {len(final_results)}")

    for final_rank, (document, score) in enumerate(final_results, 1):
        print(f"\nFinal rank {final_rank} | RRF score: {score:.6f}")
        print("-" * 60)
        print(document.page_content)
        if document.metadata:
            print("Metadata:", document.metadata)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()